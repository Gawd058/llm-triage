"""HTTP surface: OpenAI-compatible endpoints plus triage's own observability API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict
from importlib import resources
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
)

from . import __version__
from .metrics import render_metrics
from .router import Router


def _bad_request(message: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"type": "invalid_request_error", "message": message}}, status_code=400
    )


async def _read_body(request: Request) -> dict[str, Any] | Response:
    try:
        body = await request.json()
    except ValueError:
        return _bad_request("request body must be valid JSON")
    if not isinstance(body, dict):
        return _bad_request("request body must be a JSON object")
    return body


def create_app(router: Router) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        router.monitor.start()
        try:
            yield
        finally:
            await router.monitor.stop()
            await router.aclose()

    app = FastAPI(
        title="llm-triage",
        version=__version__,
        lifespan=lifespan,
        description="Routes each OpenAI-compatible request to the best available model.",
    )
    app.state.router = router

    async def route(request: Request, path: str) -> Response:
        body = await _read_body(request)
        if isinstance(body, Response):
            return body
        return await router.handle(path, body, request.headers)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await route(request, "/chat/completions")

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        return await route(request, "/completions")

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        data = [{"id": "auto", "object": "model", "owned_by": "llm-triage"}]
        data += [
            {"id": name, "object": "model", "owned_by": "llm-triage", "tier": s.cfg.tier}
            for name, s in router.states.items()
        ]
        return {"object": "list", "data": data}

    @app.get("/health")
    async def health() -> JSONResponse:
        breakers = {name: s.breaker.state for name, s in router.states.items()}
        healthy = any(s.view().healthy for s in router.states.values())
        return JSONResponse(
            {"status": "ok" if healthy else "unavailable", "backends": breakers},
            status_code=200 if healthy else 503,
        )

    @app.post("/triage/explain")
    async def explain(request: Request) -> Response:
        """Dry run: show where a request would go and why, without sending it."""
        body = await _read_body(request)
        if isinstance(body, Response):
            return body
        profile, decision = router.plan(body, request.headers)
        return JSONResponse(
            {
                "profile": asdict(profile),
                "decision": {
                    "action": decision.action,
                    "target": decision.target,
                    "reason": decision.reason,
                    "required_tier": decision.required_tier,
                    "utilization": decision.utilization,
                    "host_pressure": decision.pressure,
                    "excluded": decision.excluded,
                    "candidates": [asdict(c) for c in decision.ranked],
                },
            }
        )

    @app.get("/triage/status")
    async def status(limit: int = 25) -> dict[str, Any]:
        return router.status(limit=max(1, min(limit, 200)))

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(render_metrics(router), media_type="text/plain; version=0.0.4")

    @app.get("/", include_in_schema=False)
    async def index() -> RedirectResponse:
        return RedirectResponse("/dashboard")

    @app.get("/dashboard", include_in_schema=False)
    async def dashboard() -> HTMLResponse:
        html = resources.files("triage").joinpath("static/dashboard.html").read_text("utf-8")
        return HTMLResponse(html)

    return app
