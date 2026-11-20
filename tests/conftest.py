from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx

from triage.config import Config
from triage.mock_backend import create_mock_app
from triage.resources import StaticMonitor
from triage.router import Router
from triage.server import create_app


def three_tier_config(**overrides: Any) -> Config:
    """small (tier 1) / medium (tier 2) / large (tier 3), all served by in-process mocks."""
    data: dict[str, Any] = {
        "queue_timeout_s": 5,
        "policy": {"latency_slo_s": 10},
        "breaker": {"failure_threshold": 2, "cooldown_s": 30},
        "backends": [
            {
                "name": "small",
                "url": "http://small.mock/v1",
                "model": "m-small",
                "tier": 1,
                "max_concurrency": 4,
                "weight": 0.3,
                "prior_s_per_token": 0.004,
            },
            {
                "name": "medium",
                "url": "http://medium.mock/v1",
                "model": "m-medium",
                "tier": 2,
                "max_concurrency": 2,
                "weight": 0.6,
                "prior_s_per_token": 0.012,
            },
            {
                "name": "large",
                "url": "http://large.mock/v1",
                "model": "m-large",
                "tier": 3,
                "max_concurrency": 1,
                "weight": 1.0,
                "prior_s_per_token": 0.03,
            },
        ],
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    return Config.model_validate(data)


@asynccontextmanager
async def stack(
    cfg: Config | None = None,
    mock_options: dict[str, dict[str, Any]] | None = None,
    monitor: StaticMonitor | None = None,
) -> AsyncIterator[tuple[httpx.AsyncClient, Router, dict[str, Any]]]:
    """Router + mock backends wired together in-process, no sockets."""
    cfg = cfg or three_tier_config()
    mock_options = mock_options or {}
    mocks = {
        b.name: create_mock_app(b.name, time_scale=0.0, seed=1, **mock_options.get(b.name, {}))
        for b in cfg.backends
    }
    upstream = httpx.AsyncClient(
        mounts={f"http://{name}.mock": httpx.ASGITransport(app=app) for name, app in mocks.items()}
    )
    router = Router(cfg, monitor or StaticMonitor(), upstream, seed=1)
    app = create_app(router)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://triage"
    ) as client:
        yield client, router, mocks
    await upstream.aclose()


EASY = "What is the capital of France?"
HARD = (
    "Debug this function step by step and explain why it fails:\n"
    "```python\ndef mean(xs):\n    return sum(xs) / len(xs)\n```"
)


def chat(prompt: str, **extra: Any) -> dict[str, Any]:
    return {"model": "auto", "messages": [{"role": "user", "content": prompt}], **extra}
