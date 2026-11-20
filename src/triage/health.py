"""Live per-backend state: concurrency slots, latency estimates and a circuit breaker."""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from collections.abc import Callable, Sequence
from typing import Any

from .config import BackendConfig, BreakerConfig
from .policy import BackendView


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile; 0.0 for an empty sequence."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = math.ceil(q / 100 * len(ordered)) - 1
    return ordered[max(0, min(len(ordered) - 1, k))]


class CircuitBreaker:
    """Stops sending traffic to a failing backend, then probes it again after a cooldown.

    closed --N consecutive failures--> open --cooldown elapsed--> half_open
    half_open --probe succeeds--> closed
    half_open --probe fails--> open
    """

    def __init__(self, cfg: BreakerConfig, clock: Callable[[], float] = time.monotonic) -> None:
        self.cfg = cfg
        self.clock = clock
        self.state = "closed"
        self.consecutive_failures = 0
        self.opened_at = 0.0
        self.times_opened = 0

    def allows(self, inflight: int) -> bool:
        if self.state == "open" and self.clock() - self.opened_at >= self.cfg.cooldown_s:
            self.state = "half_open"
        if self.state == "open":
            return False
        if self.state == "half_open":
            return inflight == 0  # one probe at a time
        return True

    def on_success(self) -> None:
        self.consecutive_failures = 0
        self.state = "closed"

    def on_failure(self) -> None:
        self.consecutive_failures += 1
        tripped = self.consecutive_failures >= self.cfg.failure_threshold
        if self.state == "half_open" or tripped:
            if self.state != "open":
                self.times_opened += 1
            self.state = "open"
            self.opened_at = self.clock()


class BackendState:
    """Everything the router knows about one backend while it is running."""

    def __init__(
        self,
        cfg: BackendConfig,
        breaker: BreakerConfig,
        alpha: float = 0.3,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = cfg
        self.alpha = alpha
        self.breaker = CircuitBreaker(breaker, clock)
        self._slots = asyncio.Semaphore(cfg.max_concurrency)
        self.inflight = 0
        self.queued = 0
        self.s_per_token = cfg.prior_s_per_token
        self.ewma_latency_s = 0.0
        self.recent: deque[float] = deque(maxlen=256)
        self.successes = 0
        self.failures = 0
        self.tokens_out = 0
        self.cost_usd = 0.0

    async def acquire(self, timeout: float) -> None:
        """Wait for a concurrency slot; raises asyncio.TimeoutError after ``timeout``."""
        self.queued += 1
        try:
            await asyncio.wait_for(self._slots.acquire(), timeout)
        finally:
            self.queued -= 1
        self.inflight += 1

    def release(self) -> None:
        self.inflight -= 1
        self._slots.release()

    def record_success(self, latency_s: float, prompt_tokens: int, completion_tokens: int) -> None:
        self.successes += 1
        self.breaker.on_success()
        self.recent.append(latency_s)
        a = self.alpha
        if self.successes == 1:
            self.ewma_latency_s = latency_s
        else:
            self.ewma_latency_s = a * latency_s + (1 - a) * self.ewma_latency_s
        per_token = latency_s / max(1, completion_tokens)
        self.s_per_token = a * per_token + (1 - a) * self.s_per_token
        self.tokens_out += completion_tokens
        self.cost_usd += self.cfg.cost_per_1k_tokens * (prompt_tokens + completion_tokens) / 1000

    def record_failure(self) -> None:
        self.failures += 1
        self.breaker.on_failure()

    def view(self) -> BackendView:
        return BackendView(
            name=self.cfg.name,
            tier=self.cfg.tier,
            max_concurrency=self.cfg.max_concurrency,
            inflight=self.inflight,
            queued=self.queued,
            s_per_token=self.s_per_token,
            ewma_latency_s=self.ewma_latency_s,
            healthy=self.breaker.allows(self.inflight),
            local=self.cfg.local,
            weight=self.cfg.weight,
            cost_per_1k_tokens=self.cfg.cost_per_1k_tokens,
        )

    def snapshot(self) -> dict[str, Any]:
        recent = list(self.recent)
        return {
            "name": self.cfg.name,
            "model": self.cfg.model,
            "tier": self.cfg.tier,
            "local": self.cfg.local,
            "max_concurrency": self.cfg.max_concurrency,
            "inflight": self.inflight,
            "queued": self.queued,
            "utilization": round((self.inflight + self.queued) / self.cfg.max_concurrency, 3),
            "breaker": self.breaker.state,
            "consecutive_failures": self.breaker.consecutive_failures,
            "s_per_token": round(self.s_per_token, 5),
            "ewma_latency_s": round(self.ewma_latency_s, 4),
            "p50_latency_s": round(percentile(recent, 50), 4),
            "p95_latency_s": round(percentile(recent, 95), 4),
            "successes": self.successes,
            "failures": self.failures,
            "tokens_out": self.tokens_out,
            "cost_usd": round(self.cost_usd, 6),
        }
