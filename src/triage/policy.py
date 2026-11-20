"""Routing policy.

``decide`` is a pure function: request profile + one snapshot per backend + one
host sample in, ``Decision`` out. It does no I/O and reads no clock, so every
branch can be unit-tested with made-up numbers.

Each healthy backend b gets a score (lower is better). R is the tier the request
needs, P_q and P_l depend on request priority:

    quality  = w_quality  * P_q * max(0, R - tier_b)      model too weak for the prompt
    overkill = w_overkill * max(0, tier_b - R)            big model wasted on an easy prompt
    latency  = w_latency  * P_l * predicted_latency_b / latency_slo_s
    pressure = w_pressure * host_pressure * weight_b      only for backends on this machine
    cost     = w_cost     * expected_cost_b / cost_ref_usd

predicted_latency_b = queue wait (from in-flight + queued requests and the measured
EWMA latency) + service time (measured seconds-per-token * expected output tokens).
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from .classifier import RequestProfile
from .config import PolicyConfig
from .resources import HostSample

PRIORITY_QUALITY = {"low": 0.5, "normal": 1.0, "high": 2.0}
PRIORITY_LATENCY = {"low": 0.5, "normal": 1.0, "high": 1.5}


@dataclass(frozen=True)
class BackendView:
    """A read-only snapshot of one backend, as the policy sees it."""

    name: str
    tier: int
    max_concurrency: int
    inflight: int = 0
    queued: int = 0
    s_per_token: float = 0.02
    ewma_latency_s: float = 0.0
    healthy: bool = True
    local: bool = True
    weight: float = 1.0
    cost_per_1k_tokens: float = 0.0

    @property
    def utilization(self) -> float:
        return (self.inflight + self.queued) / self.max_concurrency

    def predicted_latency(self, output_tokens: int) -> float:
        service = self.s_per_token * output_tokens
        ahead = self.inflight + self.queued + 1 - self.max_concurrency
        if ahead <= 0:
            return service
        typical = self.ewma_latency_s or service
        return service + ahead / self.max_concurrency * typical


@dataclass(frozen=True)
class Candidate:
    name: str
    tier: int
    score: float
    predicted_latency_s: float
    terms: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    action: Literal["route", "shed"]
    reason: str
    required_tier: int
    ranked: tuple[Candidate, ...] = ()
    excluded: dict[str, str] = field(default_factory=dict)
    utilization: float = 0.0
    pressure: float = 0.0

    @property
    def target(self) -> str | None:
        return self.ranked[0].name if self.ranked else None


def required_tier(difficulty: float, tiers: Sequence[int], thresholds: Sequence[float]) -> int:
    """Map difficulty onto the configured tier ladder using the threshold cut points."""
    ladder = sorted(set(tiers))
    index = sum(1 for t in sorted(thresholds) if difficulty >= t)
    return ladder[min(index, len(ladder) - 1)]


def host_pressure(host: HostSample, cfg: PolicyConfig) -> float:
    """0 when the machine is comfortable, 1 when CPU is pegged or RAM is nearly gone."""
    cpu = (host.cpu_percent - cfg.cpu_pressure_start) / (100.0 - cfg.cpu_pressure_start)
    ram = (cfg.ram_pressure_start_mb - host.ram_available_mb) / cfg.ram_pressure_start_mb
    return round(max(0.0, min(1.0, max(cpu, ram))), 3)


def decide(
    profile: RequestProfile,
    backends: Sequence[BackendView],
    host: HostSample,
    cfg: PolicyConfig,
    *,
    rotation: int = 0,
    rng: random.Random | None = None,
) -> Decision:
    if not backends:
        return Decision(action="shed", reason="no backends configured", required_tier=1)

    excluded = {b.name: "circuit-open" for b in backends if not b.healthy}
    healthy = [b for b in backends if b.healthy]
    need = required_tier(profile.difficulty, [b.tier for b in backends], cfg.tier_thresholds)
    if profile.min_tier is not None:
        need = max(need, profile.min_tier)
    if not healthy:
        return Decision(
            action="shed",
            reason="no healthy backend (all circuits open)",
            required_tier=need,
            excluded=excluded,
        )

    capacity = sum(b.max_concurrency for b in healthy)
    utilization = round(sum(b.inflight + b.queued for b in healthy) / capacity, 3)
    pressure = host_pressure(host, cfg)
    common = {
        "required_tier": need,
        "excluded": excluded,
        "utilization": utilization,
        "pressure": pressure,
    }

    if cfg.strategy != "triage":
        ranked = _baseline(cfg.strategy, profile, healthy, rotation, rng or random.Random())
        return Decision(action="route", reason=f"strategy={cfg.strategy}", ranked=ranked, **common)

    # Admission control: when the fleet is saturated, refuse cheap work early so
    # important requests are not stuck behind it.
    limit = {"low": cfg.shed_low_at, "normal": cfg.shed_normal_at}.get(profile.priority)
    if limit is not None and utilization >= limit:
        reason = (
            f"overloaded: utilization {utilization:.2f} >= {limit:.2f} "
            f"for {profile.priority} priority"
        )
        return Decision(action="shed", reason=reason, **common)

    ranked = _score(profile, healthy, need, pressure, cfg)
    if profile.pinned:
        pinned = [c for c in ranked if c.name == profile.pinned]
        if pinned:
            ranked = pinned + [c for c in ranked if c.name != profile.pinned]
            reason = f"pinned to {profile.pinned} by request"
        else:
            reason = f"pinned backend {profile.pinned} unavailable; " + _explain(
                ranked, need, pressure
            )
    else:
        reason = _explain(ranked, need, pressure)
    return Decision(action="route", reason=reason, ranked=tuple(ranked), **common)


def _score(
    profile: RequestProfile,
    healthy: Sequence[BackendView],
    need: int,
    pressure: float,
    cfg: PolicyConfig,
) -> list[Candidate]:
    w = cfg.weights
    quality_weight = w.quality * PRIORITY_QUALITY.get(profile.priority, 1.0)
    latency_weight = w.latency * PRIORITY_LATENCY.get(profile.priority, 1.0)
    tokens = profile.prompt_tokens + profile.expected_output_tokens

    candidates = []
    for b in healthy:
        latency = b.predicted_latency(profile.expected_output_tokens)
        terms = {
            "quality": quality_weight * max(0, need - b.tier),
            "overkill": w.overkill * max(0, b.tier - need),
            "latency": latency_weight * latency / cfg.latency_slo_s,
            "pressure": w.pressure * pressure * b.weight if b.local else 0.0,
            "cost": w.cost * (b.cost_per_1k_tokens * tokens / 1000) / cfg.cost_ref_usd,
        }
        terms = {k: round(v, 4) for k, v in terms.items()}
        candidates.append(
            Candidate(b.name, b.tier, round(sum(terms.values()), 4), round(latency, 3), terms)
        )
    # Lowest score wins; on a tie prefer the stronger model, then the name for determinism.
    candidates.sort(key=lambda c: (c.score, -c.tier, c.name))
    return candidates


def _explain(ranked: Sequence[Candidate], need: int, pressure: float) -> str:
    best = ranked[0]
    if best.tier >= need:
        text = f"tier {best.tier} meets need {need}; predicted {best.predicted_latency_s:.2f}s"
    else:
        stronger = [c for c in ranked if c.tier >= need]
        if stronger:
            s = stronger[0]
            text = (
                f"degraded to tier {best.tier} (need {need}): {s.name} scored {s.score:.2f} "
                f"vs {best.score:.2f}, predicted {s.predicted_latency_s:.2f}s "
                f"vs {best.predicted_latency_s:.2f}s"
            )
        else:
            text = f"degraded to tier {best.tier} (need {need}): no healthy backend at tier {need}+"
    if pressure >= 0.25:
        text += f"; host pressure {pressure:.2f}"
    return text


def _baseline(
    strategy: str,
    profile: RequestProfile,
    healthy: Sequence[BackendView],
    rotation: int,
    rng: random.Random,
) -> tuple[Candidate, ...]:
    """Naive strategies used as comparison points in the benchmark."""
    order = sorted(healthy, key=lambda b: b.name)
    if strategy == "round_robin":
        k = rotation % len(order)
        order = order[k:] + order[:k]
    elif strategy == "random":
        rng.shuffle(order)
    elif strategy == "least_loaded":
        order.sort(key=lambda b: (b.utilization, b.name))
    elif strategy == "strongest":
        order.sort(key=lambda b: (-b.tier, b.name))
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    tokens = profile.expected_output_tokens
    return tuple(
        Candidate(b.name, b.tier, 0.0, round(b.predicted_latency(tokens), 3)) for b in order
    )
