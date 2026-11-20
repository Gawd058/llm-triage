import asyncio

import pytest

from triage.config import BackendConfig, BreakerConfig
from triage.health import BackendState, CircuitBreaker, percentile


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_breaker_opens_after_threshold_and_recovers():
    clock = FakeClock()
    breaker = CircuitBreaker(BreakerConfig(failure_threshold=2, cooldown_s=10), clock)
    breaker.on_failure()
    assert breaker.allows(0)
    breaker.on_failure()
    assert breaker.state == "open"
    assert not breaker.allows(0)

    clock.now += 10
    assert breaker.allows(0)
    assert breaker.state == "half_open"
    assert not breaker.allows(1)  # only one probe in flight

    breaker.on_success()
    assert breaker.state == "closed"
    assert breaker.times_opened == 1


def test_failed_probe_reopens_immediately():
    clock = FakeClock()
    breaker = CircuitBreaker(BreakerConfig(failure_threshold=3, cooldown_s=5), clock)
    for _ in range(3):
        breaker.on_failure()
    clock.now += 5
    assert breaker.allows(0)
    breaker.on_failure()
    assert breaker.state == "open"
    assert not breaker.allows(0)


def _state(**kw) -> BackendState:
    cfg = BackendConfig(name="b", url="http://b/v1", model="m", tier=1, **kw)
    return BackendState(cfg, BreakerConfig(), alpha=0.5)


def test_ewma_learns_seconds_per_token():
    state = _state(prior_s_per_token=0.1)
    state.record_success(latency_s=1.0, prompt_tokens=10, completion_tokens=100)  # 0.01 s/token
    assert state.s_per_token == pytest.approx(0.055)
    assert state.ewma_latency_s == pytest.approx(1.0)
    state.record_success(latency_s=3.0, prompt_tokens=10, completion_tokens=100)
    assert state.ewma_latency_s == pytest.approx(2.0)


def test_cost_is_accumulated():
    state = _state(cost_per_1k_tokens=0.5)
    state.record_success(1.0, prompt_tokens=1000, completion_tokens=1000)
    assert state.cost_usd == pytest.approx(1.0)


async def test_slots_queue_and_time_out():
    state = _state(max_concurrency=1)
    await state.acquire(1)
    assert state.inflight == 1

    with pytest.raises(asyncio.TimeoutError):
        await state.acquire(0.01)
    assert state.queued == 0  # a timed-out waiter does not leak queue depth

    waiter = asyncio.create_task(state.acquire(1))
    await asyncio.sleep(0)
    assert state.queued == 1
    state.release()
    await waiter
    assert (state.inflight, state.queued) == (1, 0)
    state.release()


def test_view_reflects_breaker():
    state = _state()
    for _ in range(3):
        state.record_failure()
    assert not state.view().healthy
    assert state.snapshot()["breaker"] == "open"


def test_percentile():
    assert percentile([], 95) == 0.0
    assert percentile([1, 2, 3, 4], 50) == 2
    assert percentile(list(range(1, 101)), 95) == 95
