"""Run configuration, loaded from YAML.

The LLM ENDPOINT (LLM_BASE_URL / LLM_API_KEY / LLM_MODEL) is not here: it comes only from env vars.
Model-dependent request parameters are here (`llm:`, resolved by llm/chat_params.py: config > env >
auto by model name), as are the price table (`prices:`, harness/cost.py) and the matrix pre-flight
cap (`max_dollars`, scripts/run_matrix.py). All three are ordinary RunConfig fields, so the baseline
drift check (harness/baselines.py) forces them to be identical in every arm.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from harness.cost import ModelPrice
from harness.fatigue import FatigueConfig

TokenParam = Literal["auto", "max_tokens", "max_completion_tokens"]
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]


class LLMSettings(BaseModel):
    """Request parameters, identical for agent and sleep calls (llm/chat_params.py).

    null = fall back to the env var (LLM_TOKEN_PARAM, LLM_REASONING_EFFORT, LLM_TEMPERATURE,
    LLM_REASONING_MODEL), then to auto-detection by model name."""

    model_config = {"extra": "forbid"}

    token_param: TokenParam | None = Field(
        None, description="auto: max_completion_tokens for reasoning models (gpt-5*, o-series...), else max_tokens")
    reasoning_effort: ReasoningEffort | None = Field(
        None, description="sent only to reasoning models; null = not sent (the model's default)")
    temperature: float | None = Field(
        None, description="sent only when the model accepts it (not to reasoning models unless effort is none)")
    reasoning_model: bool | None = Field(None, description="null = auto-detect from the model name")
    thinking: bool | None = Field(
        None, description="vLLM/Qwen only: chat_template_kwargs enable_thinking; never sent to reasoning models")


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
    llm: LLMSettings = Field(default_factory=LLMSettings)
    prices: dict[str, ModelPrice] = Field(
        default_factory=dict, description="USD per 1M tokens per model, over harness/cost.py DEFAULT_PRICES")
    max_dollars: float | None = Field(
        None, ge=0, description="run_matrix pre-flight cap on the conservative cost estimate (CLI --max-dollars wins)")
    max_evaluations: int | None = Field(
        None, gt=0, description="shared cap on evaluator calls (agent steps + replays + sandbox evals); "
                                "None = uncapped. The eval-matched counterpart of total_token_budget")
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
