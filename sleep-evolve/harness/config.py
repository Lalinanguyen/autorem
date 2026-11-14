"""Run configuration, loaded from YAML. LLM endpoint settings are NOT here: they come only from env vars."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from harness.fatigue import FatigueConfig


class ServeConfig(BaseModel):
    """Settings for infra/modal_vllm.py (the optional self-hosted endpoint)."""

    model: str = "Qwen/Qwen3-8B"
    gpu: str = "H100:1"
    vllm_version: str = "0.32.0"
    max_model_len: int = 32768


class MetricsConfig(BaseModel):
    """Per-run summary metrics (harness/metrics.py), written to summary.json["metrics"]."""

    model_config = {"extra": "forbid"}

    checkpoints: list[float] = Field(
        default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 1.0],
        description="budget fractions at which the best-so-far normalized score is reported")
    thresholds: list[float] = Field(
        default_factory=lambda: [0.95, 0.99],
        description="normalized-score targets: fraction of budget used to first reach each")
    window: int = Field(5, gt=0, description="agent steps before / after each sleep for behavior rates")


class RunConfig(BaseModel):
    problem_id: int
    seed: int = 0
    total_token_budget: int = Field(gt=0)
    max_steps: int = Field(gt=0)
    k_recent: int = Field(5, ge=0, description="last K steps shown to the agent")
    max_completion_tokens: int = 4096
    context_cap_tokens: int | None = Field(
        8000, gt=0, description="cap on the agent prompt (local counter, harness/context.py); None = uncapped"
    )
    awake_tags: bool | None = Field(
        None, description="ask the agent for the APPROACH_TAGS/... header; None = the policy decides (default off)"
    )
    sleep_policy: str = "no_sleep"
    sleep_policies: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="per-policy params; the selected policy's entry is used"
    )
    fatigue: FatigueConfig = Field(
        default_factory=FatigueConfig,
        description="fatigue monitor (harness/fatigue.py): computed and logged for every arm; policies only read it",
    )
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    baseline_arm: str | None = Field(
        None, description="baseline-suite arm name (harness/baselines.py); names the run and the summary's arm")
    eval_backend: Literal["local", "modal"] = "local"
    eval_timeout_s: int | None = None  # override the evaluator's default
    model: str | None = Field(
        None,
        description="Expected model name. The harness calls LLM_MODEL; a mismatch is logged as a warning.",
    )
    serve: ServeConfig = Field(default_factory=ServeConfig)
    out_dir: str = "runs"
    run_id: str | None = None

    @property
    def policy_params(self) -> dict[str, Any]:
        return dict(self.sleep_policies.get(self.sleep_policy, {}))

    def resolved_run_id(self) -> str:
        return self.run_id or f"p{self.problem_id}_{self.baseline_arm or self.sleep_policy}_s{self.seed}"


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env reader (KEY=VALUE lines). Values already in the environment win."""
    import os

    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_config(path: str | Path, **overrides: Any) -> RunConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    data.update({k: v for k, v in overrides.items() if v is not None})
    return RunConfig(**data)
