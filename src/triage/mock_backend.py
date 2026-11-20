"""A fake OpenAI-compatible LLM server for demos, tests and benchmarks.

It never runs a model. It has a fixed number of slots (like llama.cpp's
``--parallel``), takes ``overhead_s + tokens * s_per_token`` per request, and can
fail a set fraction of requests. That is enough to study routing behaviour
without a GPU.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import random
import time
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

WORDS = ("this", "is", "a", "simulated", "reply", "from", "a", "mock", "backend")


def create_mock_app(
    name: str,
    *,
    s_per_token: float = 0.01,
    overhead_s: float = 0.05,
    slots: int = 4,
    fail_rate: float = 0.0,
    time_scale: float = 1.0,
    default_tokens: int = 128,
    seed: int | None = None,
) -> FastAPI:
    app = FastAPI(title=f"mock-llm:{name}")
    slot_pool = asyncio.Semaphore(slots)
    rng = random.Random(seed)
    ids = itertools.count(1)
    stats = {"served": 0, "failed": 0, "active": 0, "peak_active": 0}
    app.state.stats = stats

    def error(status: int, message: str) -> JSONResponse:
        return JSONResponse({"error": {"message": message, "type": "mock_error"}}, status)

    def output_tokens(body: dict[str, Any]) -> int:
        value = body.get("max_completion_tokens") or body.get("max_tokens") or default_tokens
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return default_tokens

    def enter() -> None:
        stats["active"] += 1
        stats["peak_active"] = max(stats["peak_active"], stats["active"])

    def leave() -> None:
        stats["active"] -= 1

    def sse(payload: dict[str, Any] | str) -> bytes:
        data = payload if isinstance(payload, str) else json.dumps(payload)
        return f"data: {data}\n\n".encode()

    async def stream(n: int, model: str, chat: bool) -> AsyncIterator[bytes]:
        completion_id = f"mock-{name}-{next(ids)}"
        async with slot_pool:
            enter()
            try:
                await asyncio.sleep(overhead_s * time_scale)
                for i in range(n):
                    await asyncio.sleep(s_per_token * time_scale)
                    word = f"[{name}]" if i == 0 else WORDS[i % len(WORDS)]
                    piece = word + " "
                    choice = {"index": 0, "finish_reason": None}
                    choice.update({"delta": {"content": piece}} if chat else {"text": piece})
                    yield sse({"id": completion_id, "model": model, "choices": [choice]})
                done = {"index": 0, "finish_reason": "stop"}
                done.update({"delta": {}} if chat else {"text": ""})
                yield sse({"id": completion_id, "model": model, "choices": [done]})
                yield sse("[DONE]")
            finally:
                leave()
        stats["served"] += 1

    async def complete(request: Request, chat: bool) -> Response:
        try:
            body = await request.json()
        except ValueError:
            return error(400, "invalid JSON")
        if not isinstance(body, dict):
            return error(400, "body must be a JSON object")
        if chat and not isinstance(body.get("messages"), list):
            return error(400, "'messages' must be a list")
        if not chat and "prompt" not in body:
            return error(400, "'prompt' is required")
        if rng.random() < fail_rate:
            stats["failed"] += 1
            return error(500, f"{name}: simulated failure")

        n = output_tokens(body)
        model = str(body.get("model", name))
        prompt_tokens = max(1, len(json.dumps(body.get("messages") or body.get("prompt"))) // 4)
        if body.get("stream"):
            return StreamingResponse(stream(n, model, chat), media_type="text/event-stream")

        async with slot_pool:
            enter()
            try:
                await asyncio.sleep((overhead_s + s_per_token * n) * time_scale)
            finally:
                leave()
        stats["served"] += 1
        text = f"[{name}] simulated reply ({n} tokens)."
        choice: dict[str, Any] = {"index": 0, "finish_reason": "stop"}
        if chat:
            choice["message"] = {"role": "assistant", "content": text}
        else:
            choice["text"] = text
        return JSONResponse(
            {
                "id": f"mock-{name}-{next(ids)}",
                "object": "chat.completion" if chat else "text_completion",
                "created": int(time.time()),
                "model": model,
                "choices": [choice],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": n,
                    "total_tokens": prompt_tokens + n,
                },
            }
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await complete(request, chat=True)

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        return await complete(request, chat=False)

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": name, "object": "model", "owned_by": "mock"}]}

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", **stats}

    return app
