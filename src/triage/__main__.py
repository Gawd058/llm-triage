"""Command-line entry point: ``triage serve | demo | mock | bench | classify``."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
from pathlib import Path

from . import __version__

DEMO_BACKENDS = (
    # name, tier, port, s_per_token, overhead_s, slots, weight
    ("small", 1, 9101, 0.004, 0.05, 8, 0.3),
    ("medium", 2, 9102, 0.012, 0.10, 4, 0.6),
    ("large", 3, 9103, 0.030, 0.30, 2, 1.0),
)


def _cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .config import load_config
    from .router import Router
    from .server import create_app

    cfg = load_config(args.config)
    host, port = args.host or cfg.host, args.port or cfg.port
    print(f"llm-triage {__version__} on http://{host}:{port}  (dashboard: /dashboard)")
    uvicorn.run(create_app(Router(cfg)), host=host, port=port, log_level=args.log_level)


def _cmd_mock(args: argparse.Namespace) -> None:
    import uvicorn

    from .mock_backend import create_mock_app

    app = create_mock_app(
        args.name,
        s_per_token=args.s_per_token,
        overhead_s=args.overhead,
        slots=args.slots,
        fail_rate=args.fail_rate,
        time_scale=args.time_scale,
    )
    print(f"mock backend '{args.name}' on http://{args.host}:{args.port}/v1")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


async def _serve_all(servers: list) -> None:
    await asyncio.gather(*(server.serve() for server in servers))


def _cmd_demo(args: argparse.Namespace) -> None:
    import uvicorn

    from .config import Config
    from .mock_backend import create_mock_app
    from .router import Router
    from .server import create_app

    servers = []
    backends = []
    for name, tier, port, spt, overhead, slots, weight in DEMO_BACKENDS:
        app = create_mock_app(
            name, s_per_token=spt, overhead_s=overhead, slots=slots, time_scale=args.time_scale
        )
        servers.append(
            uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        )
        backends.append(
            {
                "name": name,
                "url": f"http://127.0.0.1:{port}/v1",
                "model": f"mock-{name}",
                "tier": tier,
                "max_concurrency": slots,
                "weight": weight,
                "prior_s_per_token": spt * args.time_scale,
            }
        )
    cfg = Config.model_validate(
        {"port": args.port, "policy": {"latency_slo_s": 10 * args.time_scale}, "backends": backends}
    )
    router_app = create_app(Router(cfg))
    servers.append(
        uvicorn.Server(
            uvicorn.Config(router_app, host="127.0.0.1", port=args.port, log_level="warning")
        )
    )
    base = f"http://127.0.0.1:{args.port}"
    print(f"llm-triage demo: 3 mock backends (ports 9101-9103) behind {base}")
    print(f"  dashboard   {base}/dashboard")
    print(f"  metrics     {base}/metrics")
    print("  try         python examples/load.py   (in another terminal)")
    print("Ctrl+C to stop.")
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_serve_all(servers))


def _cmd_bench(args: argparse.Namespace) -> None:
    from .bench import STRATEGIES, format_report, run_bench

    strategies = tuple(args.strategies.split(",")) if args.strategies else STRATEGIES
    unknown = set(strategies) - set(STRATEGIES)
    if unknown:
        sys.exit(f"unknown strategies: {sorted(unknown)}; choose from {STRATEGIES}")
    print(f"Running {len(strategies)} strategies on the same workload...")
    rows, jobs = asyncio.run(
        run_bench(args.requests, args.rate, args.seed, args.time_scale, strategies)
    )
    report = format_report(
        rows, jobs, f"rate {args.rate} req/s, seed {args.seed}, time_scale {args.time_scale}"
    )
    print()
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")


def _cmd_classify(args: argparse.Namespace) -> None:
    from dataclasses import asdict

    from .classifier import profile_request
    from .config import PolicyConfig
    from .policy import required_tier

    profile = profile_request({"messages": [{"role": "user", "content": args.text}]})
    thresholds = PolicyConfig().tier_thresholds
    out = asdict(profile) | {
        "required_tier_of_3": required_tier(profile.difficulty, [1, 2, 3], thresholds)
    }
    print(json.dumps(out, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="triage", description="Route each LLM request to the right model."
    )
    parser.add_argument("--version", action="version", version=f"llm-triage {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the router with a YAML config")
    serve.add_argument("-c", "--config", required=True)
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(func=_cmd_serve)

    demo = sub.add_parser("demo", help="router + 3 simulated backends, no GPU needed")
    demo.add_argument("--port", type=int, default=8080)
    demo.add_argument(
        "--time-scale",
        type=float,
        default=0.25,
        help="multiply simulated latencies (default 0.25 = 4x faster)",
    )
    demo.set_defaults(func=_cmd_demo)

    mock = sub.add_parser("mock", help="run one simulated OpenAI-compatible backend")
    mock.add_argument("--name", default="mock")
    mock.add_argument("--host", default="127.0.0.1")
    mock.add_argument("--port", type=int, default=9101)
    mock.add_argument("--s-per-token", type=float, default=0.01)
    mock.add_argument("--overhead", type=float, default=0.05)
    mock.add_argument("--slots", type=int, default=4)
    mock.add_argument("--fail-rate", type=float, default=0.0)
    mock.add_argument("--time-scale", type=float, default=1.0)
    mock.set_defaults(func=_cmd_mock)

    bench = sub.add_parser("bench", help="compare routing strategies on a simulated workload")
    bench.add_argument("--requests", type=int, default=240)
    bench.add_argument("--rate", type=float, default=0.8, help="base arrivals per simulated s")
    bench.add_argument("--seed", type=int, default=7)
    bench.add_argument("--time-scale", type=float, default=0.1)
    bench.add_argument("--strategies", help="comma-separated subset of strategies")
    bench.add_argument("--out", help="also write the markdown report to this file")
    bench.set_defaults(func=_cmd_bench)

    classify = sub.add_parser("classify", help="show how a prompt would be profiled")
    classify.add_argument("text")
    classify.set_defaults(func=_cmd_classify)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
