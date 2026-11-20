"""Send a mixed, bursty stream of requests to a running llm-triage router.

triage demo                 # terminal 1
python examples/load.py     # terminal 2, then watch http://127.0.0.1:8080/dashboard
"""

from __future__ import annotations

import argparse
import asyncio
import time

import httpx

from triage.bench import make_workload


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--requests", type=int, default=60)
    parser.add_argument("--rate", type=float, default=3.0, help="requests per second")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    jobs = make_workload(args.requests, args.rate, args.seed)
    start = time.perf_counter()

    async with httpx.AsyncClient(base_url=args.url, timeout=120) as client:

        async def send(job) -> None:
            await asyncio.sleep(max(0.0, job.at - (time.perf_counter() - start)))
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
            backend = response.headers.get("x-triage-backend", "-")
            difficulty = response.headers.get("x-triage-difficulty", "-")
            print(
                f"{response.status_code}  {job.priority:<6}  difficulty {difficulty:<5}  "
                f"-> {backend:<7} {time.perf_counter() - began:6.2f}s  {job.prompt[:48]!r}"
            )

        await asyncio.gather(*(send(job) for job in jobs))


if __name__ == "__main__":
    asyncio.run(main())
