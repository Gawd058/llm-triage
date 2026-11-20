"""Prometheus text exposition, rendered by hand to avoid an extra dependency."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from .policy import host_pressure

if TYPE_CHECKING:
    from .router import Router

BREAKER_CODES = {"closed": 0, "half_open": 1, "open": 2}


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _block(
    name: str, kind: str, help_text: str, samples: Iterable[tuple[dict[str, str], float]]
) -> list[str]:
    lines = [f"# HELP {name} {help_text}", f"# TYPE {name} {kind}"]
    for labels, value in samples:
        if labels:
            label_text = ",".join(f'{k}="{_escape(v)}"' for k, v in labels.items())
            lines.append(f"{name}{{{label_text}}} {value}")
        else:
            lines.append(f"{name} {value}")
    return lines


def render_metrics(router: Router) -> str:
    states = list(router.states.values())
    host = router.monitor.latest
    lines: list[str] = []
    lines += _block(
        "triage_requests_total",
        "counter",
        "Finished requests by backend and outcome.",
        (({"backend": b, "outcome": o}, n) for (b, o), n in sorted(router.outcomes.items())),
    )
    lines += _block(
        "triage_shed_total",
        "counter",
        "Requests rejected by admission control, by priority.",
        (({"priority": p}, n) for p, n in sorted(router.shed.items())),
    )
    lines += _block(
        "triage_backend_attempt_failures_total",
        "counter",
        "Failed attempts per backend (errors, 5xx, 429).",
        (({"backend": s.cfg.name}, s.failures) for s in states),
    )
    lines += _block(
        "triage_backend_inflight",
        "gauge",
        "Requests currently running on the backend.",
        (({"backend": s.cfg.name}, s.inflight) for s in states),
    )
    lines += _block(
        "triage_backend_queued",
        "gauge",
        "Requests waiting for a slot on the backend.",
        (({"backend": s.cfg.name}, s.queued) for s in states),
    )
    lines += _block(
        "triage_backend_latency_ewma_seconds",
        "gauge",
        "Exponentially weighted moving average of request latency.",
        (({"backend": s.cfg.name}, round(s.ewma_latency_s, 6)) for s in states),
    )
    lines += _block(
        "triage_backend_seconds_per_token",
        "gauge",
        "Learned seconds per output token, used to predict latency.",
        (({"backend": s.cfg.name}, round(s.s_per_token, 6)) for s in states),
    )
    lines += _block(
        "triage_backend_circuit_state",
        "gauge",
        "Circuit breaker state: 0 closed, 1 half-open, 2 open.",
        (({"backend": s.cfg.name}, BREAKER_CODES[s.breaker.state]) for s in states),
    )
    lines += _block("triage_host_cpu_percent", "gauge", "Host CPU use.", [({}, host.cpu_percent)])
    lines += _block(
        "triage_host_ram_available_megabytes",
        "gauge",
        "Host RAM available.",
        [({}, host.ram_available_mb)],
    )
    lines += _block(
        "triage_host_pressure",
        "gauge",
        "Combined host pressure used by the policy (0..1).",
        [({}, host_pressure(host, router.cfg.policy))],
    )
    return "\n".join(lines) + "\n"
