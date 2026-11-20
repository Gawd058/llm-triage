"""Configuration models, loaded from YAML and validated with pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Strategy = Literal["triage", "round_robin", "random", "least_loaded", "strongest"]


class BackendConfig(BaseModel):
    """One OpenAI-compatible model server that requests can be sent to."""

    name: str
    url: str = Field(description="Base URL including /v1, e.g. http://127.0.0.1:11434/v1")
    model: str = Field(description="Model name written into the upstream request body")
    tier: int = Field(ge=1, le=10, description="Capability rank; higher means a stronger model")
    max_concurrency: int = Field(default=4, ge=1, description="Requests in flight at once")
    local: bool = Field(default=True, description="Runs on this machine, so host load applies")
    weight: float = Field(default=1.0, ge=0, description="Relative CPU/RAM footprint on the host")
    cost_per_1k_tokens: float = Field(default=0.0, ge=0, description="USD per 1k tokens")
    prior_s_per_token: float = Field(
        default=0.02, gt=0, description="Latency guess used until real measurements arrive"
    )
    api_key_env: str | None = Field(
        default=None, description="Environment variable holding a bearer token for this backend"
    )
    timeout_s: float = Field(default=120.0, gt=0)


class Weights(BaseModel):
    """How much each term of the routing score matters. See policy.py."""

    quality: float = Field(default=1.0, ge=0)
    overkill: float = Field(default=0.3, ge=0)
    latency: float = Field(default=1.0, ge=0)
    pressure: float = Field(default=1.0, ge=0)
    cost: float = Field(default=0.5, ge=0)


class PolicyConfig(BaseModel):
    strategy: Strategy = "triage"
    latency_slo_s: float = Field(default=10.0, gt=0, description="Latency that costs 1 point")
    cost_ref_usd: float = Field(default=0.01, gt=0, description="Request cost that costs 1 point")
    tier_thresholds: list[float] = Field(
        default_factory=lambda: [0.25, 0.55],
        description="Difficulty cut points that map a request onto the tier ladder",
    )
    shed_low_at: float = Field(default=0.8, ge=0, description="Utilization that rejects low")
    shed_normal_at: float = Field(default=1.5, ge=0, description="Utilization that rejects normal")
    cpu_pressure_start: float = Field(default=70.0, ge=0, lt=100)
    ram_pressure_start_mb: float = Field(default=2048.0, gt=0)
    weights: Weights = Field(default_factory=Weights)


class BreakerConfig(BaseModel):
    failure_threshold: int = Field(default=3, ge=1)
    cooldown_s: float = Field(default=15.0, gt=0)


class Config(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8080
    queue_timeout_s: float = Field(default=30.0, gt=0, description="Max wait for a free slot")
    max_attempts: int = Field(default=3, ge=1, description="Backends tried before giving up")
    sample_interval_s: float = Field(default=1.0, gt=0)
    ewma_alpha: float = Field(default=0.3, gt=0, le=1)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    breaker: BreakerConfig = Field(default_factory=BreakerConfig)
    backends: list[BackendConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_backends(self) -> Config:
        names = [b.name for b in self.backends]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate backend names: {dupes}")
        if "auto" in names:
            raise ValueError('"auto" is reserved for automatic routing; rename that backend')
        return self


def load_config(path: str | Path) -> Config:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Config.model_validate(data)
