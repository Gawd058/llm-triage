import random

import pytest

from triage.classifier import RequestProfile
from triage.config import PolicyConfig
from triage.policy import BackendView, decide, host_pressure, required_tier
from triage.resources import HostSample

CALM = HostSample(cpu_percent=10, ram_available_mb=16000)


def fleet(**changes: dict) -> list[BackendView]:
    base = {
        "small": BackendView("small", 1, 4, s_per_token=0.004, weight=0.3),
        "medium": BackendView("medium", 2, 2, s_per_token=0.012, weight=0.6),
        "large": BackendView("large", 3, 1, s_per_token=0.03, weight=1.0),
    }
    for name, fields in changes.items():
        base[name] = BackendView(**{**base[name].__dict__, **fields})
    return list(base.values())


def req(difficulty: float, priority: str = "normal", **kw) -> RequestProfile:
    return RequestProfile(
        difficulty, prompt_tokens=50, expected_output_tokens=256, priority=priority, **kw
    )


def test_required_tier_follows_thresholds():
    assert required_tier(0.1, [1, 2, 3], [0.25, 0.55]) == 1
    assert required_tier(0.3, [1, 2, 3], [0.25, 0.55]) == 2
    assert required_tier(0.9, [1, 2, 3], [0.25, 0.55]) == 3
    # fewer tiers than thresholds: clamp to the strongest
    assert required_tier(0.9, [1, 5], [0.25, 0.55]) == 5


def test_easy_request_goes_to_smallest_model():
    decision = decide(req(0.05), fleet(), CALM, PolicyConfig())
    assert decision.action == "route"
    assert decision.target == "small"
    assert decision.required_tier == 1


def test_hard_request_goes_to_largest_model_when_idle():
    decision = decide(req(0.8), fleet(), CALM, PolicyConfig())
    assert decision.target == "large"
    assert "meets need 3" in decision.reason


def test_hard_request_degrades_when_large_is_saturated():
    busy = fleet(large={"inflight": 1, "queued": 3, "ewma_latency_s": 8.0})
    decision = decide(req(0.8), busy, CALM, PolicyConfig())
    assert decision.target == "medium"
    assert decision.reason.startswith("degraded to tier 2")
    assert [c.name for c in decision.ranked][-1] != "medium"  # large stays as a fallback


def test_open_circuit_is_excluded():
    decision = decide(req(0.8), fleet(large={"healthy": False}), CALM, PolicyConfig())
    assert decision.excluded == {"large": "circuit-open"}
    assert "large" not in [c.name for c in decision.ranked]


def test_all_circuits_open_sheds():
    views = fleet(**{n: {"healthy": False} for n in ("small", "medium", "large")})
    decision = decide(req(0.1), views, CALM, PolicyConfig())
    assert decision.action == "shed"
    assert "no healthy backend" in decision.reason


@pytest.mark.parametrize(
    ("priority", "expected"), [("low", "shed"), ("normal", "route"), ("high", "route")]
)
def test_admission_control_by_priority(priority, expected):
    # 6 of 7 slots busy -> utilization 0.857 (above shed_low_at 0.8, below shed_normal_at 1.5)
    views = fleet(small={"inflight": 4}, medium={"inflight": 2})
    assert decide(req(0.1, priority), views, CALM, PolicyConfig()).action == expected


def test_high_priority_is_never_shed():
    views = fleet(small={"inflight": 4, "queued": 40})
    assert decide(req(0.1, "high"), views, CALM, PolicyConfig()).action == "route"


def test_host_pressure_moves_traffic_off_heavy_local_models():
    views = [
        BackendView("local-large", 3, 2, s_per_token=0.03, weight=1.0, local=True),
        BackendView("cloud-large", 3, 16, s_per_token=0.03, local=False, cost_per_1k_tokens=0.002),
    ]
    cfg = PolicyConfig()
    calm = decide(req(0.8), views, CALM, cfg)
    hot = decide(req(0.8), views, HostSample(cpu_percent=98, ram_available_mb=16000), cfg)
    assert calm.target == "local-large"  # free beats paid when the machine is idle
    assert hot.target == "cloud-large"
    assert "host pressure" in hot.reason


def test_ram_pressure():
    cfg = PolicyConfig(ram_pressure_start_mb=2000)
    assert host_pressure(HostSample(cpu_percent=0, ram_available_mb=3000), cfg) == 0.0
    assert host_pressure(HostSample(cpu_percent=0, ram_available_mb=500), cfg) == 0.75


def test_pinned_backend_goes_first():
    decision = decide(req(0.05, pinned="large"), fleet(), CALM, PolicyConfig())
    assert decision.target == "large"
    assert decision.reason == "pinned to large by request"


def test_pinned_backend_unavailable_falls_back_to_scoring():
    views = fleet(large={"healthy": False})
    decision = decide(req(0.05, pinned="large"), views, CALM, PolicyConfig())
    assert decision.target == "small"
    assert decision.reason.startswith("pinned backend large unavailable")


def test_min_tier_overrides_classifier():
    decision = decide(req(0.05, min_tier=3), fleet(), CALM, PolicyConfig())
    assert decision.required_tier == 3
    assert decision.target == "large"


def test_predicted_latency_grows_with_queue():
    idle = BackendView("b", 1, 2, s_per_token=0.01, ewma_latency_s=1.0)
    full = BackendView("b", 1, 2, inflight=2, queued=4, s_per_token=0.01, ewma_latency_s=1.0)
    assert idle.predicted_latency(100) == pytest.approx(1.0)
    assert full.predicted_latency(100) == pytest.approx(1.0 + 5 / 2 * 1.0)


def test_score_breakdown_is_reported():
    decision = decide(req(0.8), fleet(), CALM, PolicyConfig())
    top = decision.ranked[0]
    assert set(top.terms) == {"quality", "overkill", "latency", "pressure", "cost"}
    assert top.score == pytest.approx(sum(top.terms.values()), abs=1e-3)


def test_baseline_strategies():
    views = fleet(small={"inflight": 3})
    rr = PolicyConfig(strategy="round_robin")
    assert [decide(req(0.1), views, CALM, rr, rotation=i).target for i in range(3)] == [
        "large",
        "medium",
        "small",
    ]
    assert decide(req(0.1), views, CALM, PolicyConfig(strategy="strongest")).target == "large"
    least = decide(req(0.1), views, CALM, PolicyConfig(strategy="least_loaded"))
    assert least.ranked[-1].name == "small"
    rand = decide(req(0.1), views, CALM, PolicyConfig(strategy="random"), rng=random.Random(0))
    assert {c.name for c in rand.ranked} == {"small", "medium", "large"}
