"""The router: profile a request, ask the policy where it should go, then forward it
with per-backend concurrency limits, fallback, circuit breaking and bookkeeping."""

from __future__ import annotations

import asyncio
import itertools
import os
import random
import time
from collections import Counter, deque
from collections.abc import AsyncIterator, Mapping
from dataclasses import asdict
from typing import Any

import anyio
import httpx
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.types import Receive, Scope, Send

from . import __version__
from .classifier import RequestProfile, estimate_tokens, profile_request
from .config import Config
from .health import BackendState
from .policy import Decision, decide, host_pressure
from .resources import HostMonitor, StaticMonitor


def _usage_int(usage: Any, key: str) -> int | None:
    if isinstance(usage, dict):
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _completion_text(data: Any) -> str:
    try:
        choice = data["choices"][0]
        return choice.get("message", {}).get("content") or choice.get("text") or ""
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""


class Router:
    def __init__(
        self,
        cfg: Config,
        monitor: HostMonitor | StaticMonitor | None = None,
        client: httpx.AsyncClient | None = None,
        *,
        seed: int | None = None,
        clock=time.monotonic,
    ) -> None:
        self.cfg = cfg
        self.monitor = monitor or HostMonitor(cfg.sample_interval_s)
        self.client = client or httpx.AsyncClient(
            timeout=None, limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100)
        )
        self.states = {
            b.name: BackendState(b, cfg.breaker, cfg.ewma_alpha, clock) for b in cfg.backends
        }
        self.decisions: deque[dict[str, Any]] = deque(maxlen=200)
        self.outcomes: Counter[tuple[str, str]] = Counter()
        self.shed: Counter[str] = Counter()
        self._rotation = itertools.count()
        self._ids = itertools.count(1)
        self._rng = random.Random(seed)

    async def aclose(self) -> None:
        await self.client.aclose()

    # ---------------------------------------------------------------- planning

    def plan(
        self, body: Mapping[str, Any], headers: Mapping[str, str] | None = None
    ) -> tuple[RequestProfile, Decision]:
        profile = profile_request(body, headers, self.states.keys())
        views = [state.view() for state in self.states.values()]
        decision = decide(
            profile,
            views,
            self.monitor.latest,
            self.cfg.policy,
            rotation=next(self._rotation),
            rng=self._rng,
        )
        return profile, decision

    # ---------------------------------------------------------------- handling

    async def handle(
        self, path: str, body: dict[str, Any], headers: Mapping[str, str] | None = None
    ) -> Response:
        """Route one OpenAI-style request (``path`` is e.g. "/chat/completions")."""
        profile, decision = self.plan(body, headers)
        entry = self._log(profile, decision)

        if decision.action == "shed":
            self.shed[profile.priority] += 1
            entry["outcome"] = "shed"
            status = 429 if decision.reason.startswith("overloaded") else 503
            return JSONResponse(
                {"error": {"type": "triage_overloaded", "message": decision.reason}},
                status_code=status,
                headers={**self._headers(entry), "Retry-After": "1"},
            )

        stream = bool(body.get("stream"))
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.cfg.queue_timeout_s
        attempts: list[str] = entry["attempts"]

        for candidate in decision.ranked[: self.cfg.max_attempts]:
            state = self.states[candidate.name]
            remaining = deadline - loop.time()
            if remaining <= 0:
                attempts.append(f"{candidate.name}: deadline exceeded")
                break
            try:
                await state.acquire(remaining)
            except asyncio.TimeoutError:
                attempts.append(f"{candidate.name}: queue timeout")
                continue

            handed_off = False
            try:
                if stream:
                    response, error = await self._open_stream(state, path, body, profile, entry)
                    handed_off = response is not None
                else:
                    response, error = await self._call(state, path, body, profile, entry)
            finally:
                if not handed_off:
                    state.release()
            if response is not None:
                return response
            attempts.append(f"{candidate.name}: {error}")

        entry["outcome"] = "failed"
        self.outcomes[("none", "failed")] += 1
        timed_out = bool(attempts) and all("timeout" in a or "deadline" in a for a in attempts)
        return JSONResponse(
            {
                "error": {
                    "type": "triage_upstream_failed",
                    "message": "no backend could serve the request",
                    "attempts": attempts,
                }
            },
            status_code=504 if timed_out else 502,
            headers=self._headers(entry),
        )

    async def _call(
        self,
        state: BackendState,
        path: str,
        body: dict[str, Any],
        profile: RequestProfile,
        entry: dict[str, Any],
    ) -> tuple[Response | None, str]:
        started = time.perf_counter()
        try:
            upstream = await self.client.post(
                self._url(state, path),
                json=self._payload(state, body),
                headers=self._upstream_headers(state),
                timeout=state.cfg.timeout_s,
            )
        except httpx.HTTPError as exc:
            state.record_failure()
            return None, type(exc).__name__
        latency = time.perf_counter() - started

        if upstream.status_code >= 500 or upstream.status_code == 429:
            state.record_failure()
            return None, f"HTTP {upstream.status_code}"
        if upstream.status_code >= 400:
            # The request itself is bad. Another backend will not fix it, and it says
            # nothing about this backend's health, so pass it straight back.
            self._finish(entry, state, "client_error", latency)
            return Response(
                upstream.content,
                status_code=upstream.status_code,
                media_type=upstream.headers.get("content-type"),
                headers=self._headers(entry),
            ), ""
        try:
            data = upstream.json()
        except ValueError:
            state.record_failure()
            return None, "invalid JSON from backend"

        usage = data.get("usage") if isinstance(data, dict) else None
        completion = _usage_int(usage, "completion_tokens") or estimate_tokens(
            _completion_text(data)
        )
        prompt = _usage_int(usage, "prompt_tokens") or profile.prompt_tokens
        state.record_success(latency, prompt, completion)
        self._finish(entry, state, "success", latency)
        return JSONResponse(data, headers=self._headers(entry)), ""

    async def _open_stream(
        self,
        state: BackendState,
        path: str,
        body: dict[str, Any],
        profile: RequestProfile,
        entry: dict[str, Any],
    ) -> tuple[Response | None, str]:
        started = time.perf_counter()
        request = self.client.build_request(
            "POST",
            self._url(state, path),
            json=self._payload(state, body),
            headers=self._upstream_headers(state),
            timeout=state.cfg.timeout_s,
        )
        try:
            upstream = await self.client.send(request, stream=True)
        except httpx.HTTPError as exc:
            state.record_failure()
            return None, type(exc).__name__
        if upstream.status_code >= 500 or upstream.status_code == 429:
            await upstream.aclose()
            state.record_failure()
            return None, f"HTTP {upstream.status_code}"

        # Failover is only possible before the first byte reaches the client, so
        # from here on the stream is committed to this backend.
        entry["backend"] = state.cfg.name
        lease = _StreamLease(self, state, upstream, profile, entry, started)
        return _GuardedStream(
            lease.relay(),
            lease=lease,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "text/event-stream"),
            headers=self._headers(entry),
        ), ""

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _url(state: BackendState, path: str) -> str:
        return state.cfg.url.rstrip("/") + path

    @staticmethod
    def _payload(state: BackendState, body: dict[str, Any]) -> dict[str, Any]:
        return {**body, "model": state.cfg.model}

    @staticmethod
    def _upstream_headers(state: BackendState) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if state.cfg.api_key_env:
            key = os.environ.get(state.cfg.api_key_env)
            if key:
                headers["authorization"] = f"Bearer {key}"
        return headers

    def _log(self, profile: RequestProfile, decision: Decision) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "id": next(self._ids),
            "at": round(time.time(), 3),
            "priority": profile.priority,
            "difficulty": profile.difficulty,
            "signals": list(profile.signals),
            "required_tier": decision.required_tier,
            "action": decision.action,
            "reason": decision.reason,
            "utilization": decision.utilization,
            "host_pressure": decision.pressure,
            "candidates": [asdict(c) for c in decision.ranked[:3]],
            "excluded": dict(decision.excluded),
            "backend": None,
            "outcome": "pending",
            "latency_s": None,
            "attempts": [],
        }
        self.decisions.append(entry)
        return entry

    def _finish(
        self, entry: dict[str, Any], state: BackendState, outcome: str, latency: float
    ) -> None:
        entry.update(backend=state.cfg.name, outcome=outcome, latency_s=round(latency, 4))
        self.outcomes[(state.cfg.name, outcome)] += 1

    @staticmethod
    def _headers(entry: dict[str, Any]) -> dict[str, str]:
        headers = {
            "x-triage-request-id": str(entry["id"]),
            "x-triage-difficulty": f"{entry['difficulty']:.3f}",
            "x-triage-reason": entry["reason"],
        }
        if entry["backend"]:
            headers["x-triage-backend"] = entry["backend"]
        if entry["attempts"]:
            headers["x-triage-fallbacks"] = str(len(entry["attempts"]))
        return headers

    # ---------------------------------------------------------------- status

    def status(self, limit: int = 25) -> dict[str, Any]:
        host = self.monitor.latest
        states = list(self.states.values())
        capacity = sum(s.cfg.max_concurrency for s in states)
        load = sum(s.inflight + s.queued for s in states)
        return {
            "version": __version__,
            "strategy": self.cfg.policy.strategy,
            "utilization": round(load / capacity, 3) if capacity else 0.0,
            "host": {**asdict(host), "pressure": host_pressure(host, self.cfg.policy)},
            "backends": [s.snapshot() for s in states],
            "shed": dict(self.shed),
            "recent": list(reversed(self.decisions))[:limit],
        }


class _StreamLease:
    """Gives a streamed request's slot back exactly once, however the stream ends:
    completed, upstream error, client disconnect, or never started at all."""

    def __init__(
        self,
        router: Router,
        state: BackendState,
        upstream: httpx.Response,
        profile: RequestProfile,
        entry: dict[str, Any],
        started: float,
    ) -> None:
        self.router = router
        self.state = state
        self.upstream = upstream
        self.profile = profile
        self.entry = entry
        self.started = started
        self.events = 0
        self.closed = False

    async def relay(self) -> AsyncIterator[bytes]:
        outcome = "error"
        try:
            async for chunk in self.upstream.aiter_raw():
                self.events += chunk.count(b"data:") - chunk.count(b"[DONE]")
                yield chunk
            outcome = "success" if self.upstream.status_code < 400 else "client_error"
        except httpx.HTTPError:
            outcome = "error"
        except BaseException:
            outcome = "cancelled"
            raise
        finally:
            await self.close(outcome)

    async def close(self, outcome: str) -> None:
        if self.closed:
            return
        self.closed = True
        latency = time.perf_counter() - self.started
        self.state.release()
        if outcome == "success":
            self.state.record_success(latency, self.profile.prompt_tokens, max(1, self.events))
        elif outcome == "error":
            self.state.record_failure()
        self.router._finish(self.entry, self.state, outcome, latency)
        with anyio.CancelScope(shield=True):
            await self.upstream.aclose()


class _GuardedStream(StreamingResponse):
    """A StreamingResponse that always releases its lease, even if the body never ran."""

    def __init__(self, *args: Any, lease: _StreamLease, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._lease = lease

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._lease.close("cancelled")
