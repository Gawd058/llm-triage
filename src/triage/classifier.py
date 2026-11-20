"""Cheap, explainable request profiling.

Before a request goes anywhere, triage estimates how hard it is (0..1) from
signals in the prompt. The estimate is a transparent heuristic on purpose: it
runs in microseconds, needs no model, and every point of difficulty comes from a
named signal that shows up in logs, response headers and the dashboard.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any

PRIORITIES = ("low", "normal", "high")

REASONING_HINTS = (
    "step by step",
    "step-by-step",
    "prove",
    "derive",
    "analyze",
    "analyse",
    "explain why",
    "compare",
    "trade-off",
    "tradeoff",
    "debug",
    "optimize",
    "optimise",
    "design a",
    "architecture",
    "refactor",
    "algorithm",
    "complexity",
    "evaluate",
    "critique",
)

CODE_RE = re.compile(
    r"```|\bdef \w+\(|\bclass \w+[:(]|\bfunction \w+\(|#include\b|"
    r"\bSELECT\b[^;]{1,200}?\bFROM\b|Traceback \(most recent call last\)",
    re.IGNORECASE,
)
MATH_RE = re.compile(
    r"\\frac|\\int|\\sum|\bintegral\b|\bderivative\b|\bmatrix\b|\beigen\w*|\btheorem\b|"
    r"\bequation\b|\w\s*\^\s*\d",
    re.IGNORECASE,
)

LENGTH_WEIGHT = 0.35
LENGTH_SATURATION_TOKENS = 2000
HINT_WEIGHT = 0.2
HINT_CAP = 0.5
CODE_WEIGHT = 0.25
MATH_WEIGHT = 0.2
MULTI_TURN_WEIGHT = 0.1
LONG_OUTPUT_WEIGHT = 0.1


@dataclass(frozen=True)
class RequestProfile:
    difficulty: float
    prompt_tokens: int
    expected_output_tokens: int
    priority: str = "normal"
    signals: tuple[str, ...] = ()
    pinned: str | None = None
    min_tier: int | None = None


def extract_text(body: Mapping[str, Any]) -> tuple[str, int]:
    """Return the prompt text of a chat or completions request and its message count."""
    messages = body.get("messages")
    if isinstance(messages, list):
        parts: list[str] = []
        for message in messages:
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):  # multimodal "content parts"
                parts.extend(
                    p["text"]
                    for p in content
                    if isinstance(p, dict) and isinstance(p.get("text"), str)
                )
        return "\n".join(parts), len(messages)
    prompt = body.get("prompt", "")
    if isinstance(prompt, list):
        prompt = "\n".join(str(p) for p in prompt)
    return str(prompt), 1


def estimate_tokens(text: str) -> int:
    """Roughly 4 characters per token for English text; good enough for routing."""
    return max(1, len(text) // 4)


def _max_tokens(body: Mapping[str, Any]) -> int | None:
    value = body.get("max_completion_tokens") or body.get("max_tokens")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def profile_request(
    body: Mapping[str, Any],
    headers: Mapping[str, str] | None = None,
    backend_names: Collection[str] = (),
) -> RequestProfile:
    headers = {k.lower(): v for k, v in (headers or {}).items()}
    text, n_messages = extract_text(body)
    lowered = text.lower()
    tokens = estimate_tokens(text)
    signals: list[str] = []
    score = 0.0

    score += LENGTH_WEIGHT * min(1.0, tokens / LENGTH_SATURATION_TOKENS)
    if tokens >= 100:
        signals.append(f"length:{tokens}tok")

    hints = [h for h in REASONING_HINTS if h in lowered]
    if hints:
        score += min(HINT_CAP, HINT_WEIGHT * len(hints))
        signals.append("reasoning:" + ",".join(hints[:3]))
    if CODE_RE.search(text):
        score += CODE_WEIGHT
        signals.append("code")
    if MATH_RE.search(text):
        score += MATH_WEIGHT
        signals.append("math")
    if n_messages > 6:
        score += MULTI_TURN_WEIGHT
        signals.append(f"multi-turn:{n_messages}")

    max_tokens = _max_tokens(body)
    if max_tokens is not None and max_tokens >= 1000:
        score += LONG_OUTPUT_WEIGHT
        signals.append(f"long-output:{max_tokens}")

    difficulty = round(min(1.0, score), 3)
    guess = int(96 + 400 * difficulty)
    expected_output = min(max_tokens, guess) if max_tokens else guess

    priority = headers.get("x-priority", "normal").strip().lower()
    if priority not in PRIORITIES:
        priority = "normal"

    model = body.get("model")
    pinned = model if isinstance(model, str) and model in backend_names else None

    min_tier: int | None = None
    raw_min_tier = headers.get("x-triage-min-tier")
    if raw_min_tier is not None:
        try:
            min_tier = max(1, int(raw_min_tier))
        except ValueError:
            min_tier = None

    return RequestProfile(
        difficulty=difficulty,
        prompt_tokens=tokens,
        expected_output_tokens=expected_output,
        priority=priority,
        signals=tuple(signals),
        pinned=pinned,
        min_tier=min_tier,
    )
