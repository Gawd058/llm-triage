"""Reproducible routing benchmark.

The same seeded workload is replayed against the same three simulated backends
once per routing strategy, entirely in-process (no sockets, no GPU). Only the
routing strategy changes between runs, so differences in the results come from
routing alone.

Simulated backends (latency = overhead + tokens * s_per_token):

    small   tier 1   4 ms/token    8 slots   ~0.5B model on a laptop
    medium  tier 2  12 ms/token    4 slots   ~3B model, 2% injected failures
    large   tier 3  30 ms/token    2 slots   ~7B model

To keep runs short, every sleep is multiplied by ``time_scale``; reported
latencies are divided by it again, so they are in simulated seconds.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import Counter
from dataclasses import dataclass

import httpx

from .classifier import profile_request
from .config import Config
from .health import percentile
from .mock_backend import create_mock_app
from .policy import required_tier
from .resources import StaticMonitor
from .router import Router
from .server import create_app

STRATEGIES = ("round_robin", "random", "least_loaded", "strongest", "triage")


@dataclass(frozen=True)
class SimBackend:
    name: str
    tier: int
    s_per_token: float
    overhead_s: float
    slots: int
    weight: float
    fail_rate: float


SIM_BACKENDS = (
    SimBackend("small", 1, 0.004, 0.05, 8, 0.3, 0.0),
    SimBackend("medium", 2, 0.012, 0.10, 4, 0.6, 0.02),
    SimBackend("large", 3, 0.030, 0.30, 2, 1.0, 0.0),
)
TIER_OF = {b.name: b.tier for b in SIM_BACKENDS}
MAX_TOKENS = {1: 48, 2: 192, 3: 384}

# Prompt templates by true difficulty tier. Some are deliberately ambiguous so
# the classifier is not graded only on easy cases.
TEMPLATES = {
    1: [
        "What is the capital of {country}?",
        "Translate 'good morning' into {lang}.",
        "Give me a synonym for '{word}'.",
        "Say hello in {lang}.",
        "How many days are in a leap year?",
        "Write a one-line birthday wish for my friend who likes {thing}.",
    ],
    2: [
        "Explain why {topic} matters and compare it with {topic2}.",
        "Summarize the trade-offs between {topic} and {topic2} and evaluate which fits a "
        "small team.",
        "Analyze the main causes of the {event} and evaluate which mattered most.",
        "Write a short cover letter for a junior developer role at a company that builds "
        "{thing} software.",
        "Compare REST and GraphQL for a mobile app backend.",
    ],
    3: [
        "Debug this function step by step and explain why it fails on an empty list:\n"
        "```python\ndef {fn}(xs):\n    return sum(xs) / len(xs)\n```",
        "Prove that the derivative of x^2 is 2x, step by step, from the limit definition.",
        "Design a rate limiter architecture for an API gateway and analyze the time "
        "complexity of the token-bucket algorithm:\n```python\nclass Bucket:\n    pass\n```",
        "Refactor this code and explain why the new version is faster:\n```python\n"
        "def {fn}(items):\n    out = []\n    for i in items:\n        if i not in out:\n"
        "            out.append(i)\n    return out\n```",
        "Plan a migration of a monolith with 40 services' worth of code into microservices, "
        "with a rollout schedule and risks.",
    ],
}
FILLERS = {
    "country": ["France", "Japan", "Kenya", "Peru", "Canada"],
    "lang": ["Spanish", "Hindi", "German", "Japanese"],
    "word": ["happy", "fast", "bright", "quiet"],
    "thing": ["chess", "cycling", "astronomy", "cooking"],
    "topic": ["caching", "indexing", "sharding", "replication"],
    "topic2": ["batching", "compression", "partitioning", "queueing"],
    "event": ["2008 financial crisis", "fall of Rome", "dot-com crash"],
    "fn": ["mean", "average", "dedupe", "unique"],
}


@dataclass(frozen=True)
class Job:
    at: float  # simulated seconds after start
    tier: int  # true difficulty label
    priority: str
    prompt: str
    max_tokens: int


@dataclass(frozen=True)
class Result:
    job: Job
    status: int
    latency: float  # simulated seconds
    backend: str | None


def make_workload(n: int, rate: float, seed: int, burst: float = 3.0) -> list[Job]:
    """Poisson arrivals at ``rate`` req/s, ``burst`` times faster in the middle third.
    Mix: 50% easy / 30% medium / 20% hard; 20% high / 60% normal / 20% low priority."""
    rng = random.Random(seed)
    jobs, t = [], 0.0
    for i in range(n):
        in_burst = n / 3 <= i < 2 * n / 3
        t += rng.expovariate(rate * (burst if in_burst else 1.0))
        r = rng.random()
        tier = 1 if r < 0.5 else 2 if r < 0.8 else 3
        p = rng.random()
        priority = "high" if p < 0.2 else "low" if p >= 0.8 else "normal"
        template = rng.choice(TEMPLATES[tier])
        prompt = template.format(**{k: rng.choice(v) for k, v in FILLERS.items()})
        jobs.append(Job(round(t, 4), tier, priority, prompt, MAX_TOKENS[tier]))
    return jobs


def bench_config(strategy: str, time_scale: float) -> Config:
    return Config.model_validate(
        {
            "queue_timeout_s": 30 * time_scale,
            "policy": {"strategy": strategy, "latency_slo_s": 10 * time_scale},
            "breaker": {"failure_threshold": 3, "cooldown_s": 15 * time_scale},
            "backends": [
                {
                    "name": b.name,
                    "url": f"http://{b.name}.mock/v1",
                    "model": f"sim-{b.name}",
                    "tier": b.tier,
                    "max_concurrency": b.slots,
                    "weight": b.weight,
                    "prior_s_per_token": b.s_per_token * time_scale,
                }
                for b in SIM_BACKENDS
            ],
        }
    )


async def run_strategy(
    strategy: str, jobs: list[Job], *, time_scale: float = 0.1, seed: int = 7
) -> list[Result]:
    mocks = {
        b.name: create_mock_app(
            b.name,
            s_per_token=b.s_per_token,
            overhead_s=b.overhead_s,
            slots=b.slots,
            fail_rate=b.fail_rate,
            time_scale=time_scale,
            seed=seed + i,
        )
        for i, b in enumerate(SIM_BACKENDS)
    }
    upstream = httpx.AsyncClient(
        mounts={f"http://{name}.mock": httpx.ASGITransport(app=app) for name, app in mocks.items()},
        timeout=None,
    )
    router = Router(bench_config(strategy, time_scale), StaticMonitor(), upstream, seed=seed)
    app = create_app(router)
    results: list[Result] = []
    loop = asyncio.get_running_loop()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://triage", timeout=None
    ) as client:
        start = loop.time()

        async def fire(job: Job) -> None:
            await asyncio.sleep(max(0.0, start + job.at * time_scale - loop.time()))
            began = time.perf_counter()
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": job.prompt}],
                    "max_tokens": job.max_tokens,
                },
                headers={"x-priority": job.priority},
            )
            elapsed = (time.perf_counter() - began) / time_scale
            results.append(
                Result(job, response.status_code, elapsed, response.headers.get("x-triage-backend"))
            )

        await asyncio.gather(*(fire(job) for job in jobs))
    await upstream.aclose()
    return results


def summarize(strategy: str, results: list[Result]) -> dict[str, object]:
    n = len(results)
    served = [r for r in results if r.status == 200]
    shed = sum(1 for r in results if r.status == 429)
    latencies = [r.latency for r in served]
    high = [r.latency for r in served if r.job.priority == "high"]
    quality = [TIER_OF[r.backend] >= r.job.tier for r in served if r.backend]
    shares = Counter(r.backend for r in served)
    # Why was a request answered by a model weaker than it needed?
    misses = Counter(
        "classifier" if predicted_tier(r.job.prompt) < r.job.tier else "routing"
        for r in served
        if r.backend and TIER_OF[r.backend] < r.job.tier
    )
    return {
        "strategy": strategy,
        "served": len(served) / n,
        "shed": shed / n,
        "failed": (n - len(served) - shed) / n,
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "p95_high": percentile(high, 95),
        "quality": sum(quality) / len(quality) if quality else 0.0,
        "share": "/".join(f"{shares[b.name] / max(1, len(served)):.0%}" for b in SIM_BACKENDS),
        "misses": dict(misses),
    }


def predicted_tier(prompt: str) -> int:
    """The tier triage's classifier would ask for, given the bench's backend ladder."""
    profile = profile_request({"messages": [{"role": "user", "content": prompt}]})
    thresholds = bench_config("triage", 1.0).policy.tier_thresholds
    return required_tier(profile.difficulty, [b.tier for b in SIM_BACKENDS], thresholds)


def classifier_agreement(jobs: list[Job]) -> float:
    return sum(predicted_tier(job.prompt) == job.tier for job in jobs) / len(jobs)


def format_report(rows: list[dict[str, object]], jobs: list[Job], args_text: str) -> str:
    lines = [
        f"Workload: {len(jobs)} requests, {args_text}",
        "50% easy / 30% medium / 20% hard; 20% high / 60% normal / 20% low priority;",
        "Poisson arrivals with a 3x burst in the middle third. Latencies in simulated seconds.",
        "",
        "| strategy | served | shed (429) | failed | p50 s | p95 s | p95 high-pri s "
        "| quality met | share small/medium/large |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['strategy']} | {r['served']:.1%} | {r['shed']:.1%} | {r['failed']:.1%} "
            f"| {r['p50']:.2f} | {r['p95']:.2f} | {r['p95_high']:.2f} | {r['quality']:.1%} "
            f"| {r['share']} |"
        )
    lines += [
        "",
        "quality met = share of served requests answered by a model at or above the "
        "request's true tier.",
        f"Classifier agreement with the hand labels: {classifier_agreement(jobs):.1%} "
        "(optimistic: the labels and the heuristic were written by the same person).",
    ]
    for r in rows:
        if r["strategy"] == "triage":
            misses = r["misses"]
            lines.append(
                f"triage quality misses: {misses.get('classifier', 0)} because the classifier "
                f"underestimated the prompt, {misses.get('routing', 0)} because the right "
                "model was busy and the request was degraded."
            )
    return "\n".join(lines)


async def run_bench(
    n: int = 240,
    rate: float = 0.8,
    seed: int = 7,
    time_scale: float = 0.1,
    strategies: tuple[str, ...] = STRATEGIES,
    progress: bool = True,
) -> tuple[list[dict[str, object]], list[Job]]:
    jobs = make_workload(n, rate, seed)
    rows = []
    for strategy in strategies:
        began = time.perf_counter()
        results = await run_strategy(strategy, jobs, time_scale=time_scale, seed=seed)
        rows.append(summarize(strategy, results))
        if progress:
            print(f"  {strategy:<13} done in {time.perf_counter() - began:5.1f}s wall", flush=True)
    return rows, jobs
