"""GEPA arm: `gepa.optimize_anything` on one problem, budget-matched to the evolution loop.

Matched budget (README "Budget semantics"):
  - Tokens: every GEPA LLM call (reflection/proposal) goes through infra/llm_proxy.py, which
    charges the API-reported usage to the run's `Budget` as role "agent" and refuses with
    `budget_exhausted` once the total is spent. Sleep calls (gepa_sleep arm) go through an
    `LLMClient` on the same Budget.
  - Evaluations: `max_metric_calls` (default `max_steps`) counts actual sandbox evaluations of
    distinct candidates after the seed, so step indices run 0 (seed) .. max_steps exactly as in
    harness/loop.py. GEPA's own `total_num_evals` also counts re-evaluations of an
    already-scored candidate (the parent's minibatch eval each iteration, the accepted child's
    valset eval). Those are served from a memo here: no sandbox run, no log line, no charge.

Every sandbox evaluation is one `event: "step"` line in history.jsonl (harness/logging.py
`RunLogger`, same FIELDS), and summary.json / best_program.py mirror harness/loop.py. As in the
loop, a FatigueMonitor (harness/fatigue.py, `cfg.fatigue`) observes every step line before it is
written and its fields are logged with it (GEPA prompts carry no awake header, so calibration and
perseveration are null; load, stagnation, sloppiness, narrowing still measure).

Requires the optional dependency group: `uv sync --group gepa`.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import yaml
from pydantic import BaseModel, Field

from evaluators.base import INVALID_SCORE, EvalResult, Evaluator, invalid, normalized_score
from harness.budget import Budget, BudgetExhausted
from harness.config import RunConfig
from harness.fatigue import FatigueMonitor, SleepDepth
from harness.logging import FIELDS, RunLogger
from infra.llm_proxy import LLMProxy, ProxyLM

ARM_NAMES = {"gepa": "gepa", "gepa_sleep": "gepa_full_sleep"}

DEFAULT_BACKGROUND = (
    "The component is a complete, self-contained Python program. It must define construct(), "
    "which returns the candidate solution; it is run in a fresh interpreter with a time limit "
    "and only numpy and the standard library are available. The evaluation results contain the "
    "score, whether the solution was valid, the error if any, and diagnostics (constraint "
    "violations with amounts, slack values, and an ASCII render where it applies). Return the "
    "complete improved program in a single ```python block."
)


class GEPAParams(BaseModel):
    """Arm-specific settings (the `gepa:` section of configs/gepa.yaml)."""

    max_metric_calls: int | None = Field(
        None, gt=0, description="sandbox evaluations after the seed; default = max_steps (matched)"
    )
    reflection_max_tokens: int | None = Field(None, description="default: max_completion_tokens")
    temperature: float | None = None
    candidate_selection_strategy: Literal["pareto", "current_best", "epsilon_greedy", "top_k_pareto"] = "pareto"
    invalid_score: float = Field(-1e9, description="finite score GEPA sees for invalid candidates")
    background: str = DEFAULT_BACKGROUND
    max_llm_failures: int = Field(5, description="stop after this many non-budget LLM errors in a row")
    internal_limit_factor: int = Field(
        4, description="backstop: GEPA's own max_metric_calls = factor * (max_metric_calls + 1) + 8"
    )
    # gepa_sleep arm only
    round_metric_calls: int = Field(10, gt=0, description="k: sandbox evaluations per GEPA round")
    memory_token_cap: int | None = Field(None, description="token_cap passed to memory_section()")
    sleep_policy: str = "full_sleep"


class GEPAArmConfig(RunConfig):
    """RunConfig plus the arm choice and GEPA settings. Unused loop fields (k_recent...) are ignored."""

    arm: Literal["gepa", "gepa_sleep"] = "gepa"
    gepa: GEPAParams = Field(default_factory=GEPAParams)

    @property
    def metric_calls(self) -> int:
        return self.gepa.max_metric_calls or self.max_steps

    def resolved_run_id(self) -> str:
        return self.run_id or f"p{self.problem_id}_{ARM_NAMES[self.arm]}_s{self.seed}"


def load_arm_config(path: str | Path, **overrides: Any) -> GEPAArmConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    data.update({k: v for k, v in overrides.items() if v is not None})
    return GEPAArmConfig(**data)


_PY_BLOCK = re.compile(r"```(?:python|py)[ \t]*\n(.*?)```", re.DOTALL)


def clean_candidate(text: str) -> str:
    """GEPA keeps the text between the first and last ``` fence; drop any leftover fences."""
    if "```" in text:
        blocks = _PY_BLOCK.findall(text) or re.findall(r"```[^\n]*\n(.*?)```", text, re.DOTALL)
        if blocks:
            text = blocks[-1]
        else:
            text = text.replace("```python", "").replace("```", "")
    return text.strip() + "\n"


def asi_from_result(result: EvalResult) -> dict:
    """EvalResult -> GEPA side_info (ASI).

    gepa renders nested dicts/lists as one markdown header per item
    (gepa/strategies/instruction_proposal.py, format_samples), which is very verbose, so
    nested values are serialized as compact JSON. The key "scores" is reserved by gepa for
    multi-objective tracking and is never emitted.
    """
    asi: dict[str, Any] = {
        "Score": result.score if result.valid else None,
        "Valid": result.valid,
    }
    if result.error:
        asi["Error"] = result.error
    for k, v in result.side_info.items():
        key = "diag_scores" if k == "scores" else k
        if isinstance(v, (dict, list)):
            asi[key] = json.dumps(v, separators=(",", ":"))
        else:
            asi[key] = v
    return asi


class _FileLogger:
    """gepa LoggerProtocol: append messages to a file instead of stdout."""

    def __init__(self, path: Path):
        self.path = path

    def log(self, message: str) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(str(message).rstrip() + "\n")


@dataclass
class _Best:
    program_id: str | None = None
    code: str | None = None
    score: float = INVALID_SCORE
    step: int | None = None


@dataclass
class GEPASession:
    """State shared across one or more `optimize_anything` rounds of a single run."""

    cfg: GEPAArmConfig
    evaluator: Evaluator
    budget: Budget
    lm: ProxyLM
    logger: RunLogger
    out: Path
    history: list[dict] = field(default_factory=list)  # mirror of history.jsonl
    memo: dict[str, tuple[str, EvalResult]] = field(default_factory=dict)  # code -> (program_id, result)
    best: _Best = field(default_factory=_Best)
    next_step: int = 0
    limit_step: int = 0  # stop the current round once next_step > limit_step
    current_parent: str | None = None
    current_iteration: int | None = None
    current_round: int = 0
    gepa_internal_calls: int = 0
    gepa_iterations: int = 0
    over_limit_calls: int = 0
    error: str | None = None
    observe: Callable[[int, dict], dict] | None = None  # optional sleep-policy hook
    monitor: FatigueMonitor | None = None  # fatigue, measured for every step as in harness/loop.py

    # -------------------------------------------------------------- evaluation

    @property
    def steps_done(self) -> int:
        return max(0, self.next_step - 1)

    def evaluate(self, candidate: str) -> tuple[float, dict]:
        """The evaluator handed to gepa: returns (score, side_info)."""
        code = clean_candidate(candidate)
        if code in self.memo:
            _, result = self.memo[code]
            return self._gepa_score(result), asi_from_result(result)
        if self.next_step > self.limit_step:
            # gepa checks stoppers only between iterations; one iteration adds at most one new
            # candidate, so this is a safety net that should never trigger.
            self.over_limit_calls += 1
            return self.cfg.gepa.invalid_score, {"Error": "evaluation limit reached; not evaluated"}
        try:
            result = self.evaluator.evaluate(code)
        except Exception as e:  # evaluators should not raise; never crash the run on one
            result = invalid(f"evaluator raised {type(e).__name__}: {e}")
        self._record(code, result)
        return self._gepa_score(result), asi_from_result(result)

    def _gepa_score(self, result: EvalResult) -> float:
        return result.score if result.valid and math.isfinite(result.score) else self.cfg.gepa.invalid_score

    def _record(self, code: str, result: EvalResult) -> None:
        step = self.next_step
        pid = f"p{step}"
        parent = self.memo.get(self.current_parent)[0] if self.current_parent in self.memo else None
        prev = self.best.score if self.best.program_id is not None else None
        delta = result.score - prev if (result.valid and prev is not None and math.isfinite(prev)) else None
        details = dict(result.details)
        details["gepa"] = {"iteration": self.current_iteration, "round": self.current_round}
        fields = dict(
            step=step, parent_id=parent, program_id=pid, program=code,
            score=result.score if result.valid else INVALID_SCORE, valid=result.valid,
            error=result.error, delta_vs_best=delta, event="step", details=details,
            side_info=result.side_info,
        )
        if self.observe is not None:  # sleep policy hook (sleep/full.py), as in harness/loop.py
            obs = self.observe(step, dict(fields, score=result.score if result.valid else None)) or {}
            for k in ("pressure", "perseveration_hit"):
                if k in FIELDS:
                    fields[k] = obs.get(k)
            if obs.get("perseveration_items"):
                details["perseveration_items"] = obs["perseveration_items"]
        rec = self.logger.build(agent_tokens=self.budget.agent_tokens, sleep_tokens=self.budget.sleep_tokens,
                                **fields)
        if self.monitor is not None:
            self.monitor.observe(rec)
            rec.update(self.monitor.log_fields())
        self.history.append(self.logger.write(rec))
        self.memo[code] = (pid, result)
        if result.valid and (self.best.program_id is None or result.score > self.best.score):
            self.best = _Best(pid, code, result.score, step)
        elif self.best.program_id is None and step == 0:
            self.best = _Best(pid, code, INVALID_SCORE, step)  # keep the seed as a fallback seed
        self.next_step += 1

    # -------------------------------------------------------------- stopping

    def budget_exhausted(self) -> bool:
        return self.lm.exhausted or self.budget.exhausted

    def llm_failed(self) -> bool:
        return self.lm.consecutive_failures >= self.cfg.gepa.max_llm_failures

    def should_stop(self, _state: Any = None) -> bool:
        return self.budget_exhausted() or self.llm_failed() or self.next_step > self.limit_step

    # gepa callbacks (gepa/core/callbacks.py): track the parent of the next proposal
    def on_candidate_selected(self, event: dict) -> None:
        cand = event.get("candidate") or {}
        self.current_parent = clean_candidate(next(iter(cand.values()), "")) if cand else None
        self.current_iteration = event.get("iteration")

    def on_iteration_end(self, event: dict) -> None:
        self.gepa_iterations += 1

    # -------------------------------------------------------------- one gepa round

    def optimize(self, seed_code: str, background: str, new_evals: int) -> None:
        """Run optimize_anything from `seed_code` until `new_evals` more sandbox evaluations
        (the seed's first evaluation is step 0 and does not count), or a global stop."""
        from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything

        base = max(self.next_step, 1)  # step 0 (seed) is free, as in harness/loop.py
        self.limit_step = min(self.cfg.metric_calls, base - 1 + new_evals)
        if self.next_step > 0 and self.should_stop():
            return
        self.current_parent = None
        self.current_iteration = 0
        internal = self.cfg.gepa.internal_limit_factor * (self.cfg.metric_calls + 1) + 8
        config = GEPAConfig(
            engine=EngineConfig(
                run_dir=None,
                seed=self.cfg.seed + self.current_round,
                display_progress_bar=False,
                raise_on_exception=True,
                use_cloudpickle=False,
                max_metric_calls=internal,
                candidate_selection_strategy=self.cfg.gepa.candidate_selection_strategy,
                parallel=False,
                max_workers=1,
                cache_evaluation=False,  # we memoize by program text ourselves
            ),
            reflection=ReflectionConfig(reflection_lm=self.lm),
            stop_callbacks=[self.should_stop],
            callbacks=[self],
        )
        config.tracking.logger = _FileLogger(self.out / "gepa_log.txt")
        try:
            result = optimize_anything(
                seed_candidate=seed_code,
                evaluator=self.evaluate,
                objective=self.evaluator.description,
                background=background or None,
                config=config,
            )
            self.gepa_internal_calls += int(result.total_metric_calls or 0)
        except BudgetExhausted:
            pass  # recorded by budget_exhausted(); gepa normally swallows reflection errors
        except Exception as e:  # a gepa-internal failure ends the run cleanly with outputs written
            self.error = f"{type(e).__name__}: {e}"
            config.tracking.logger.log(f"optimize_anything raised {self.error}")


def stop_reason(session: GEPASession) -> str:
    if session.budget_exhausted():
        return "budget_exhausted"
    if session.error:
        return "gepa_error"
    if session.steps_done >= session.cfg.metric_calls:
        return "max_steps"
    if session.llm_failed():
        return "llm_error"
    return "gepa_internal_limit"


def gepa_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("gepa")
    except Exception:
        return None


def open_session(cfg: GEPAArmConfig, evaluator: Evaluator, budget: Budget, proxy: LLMProxy,
                 out: Path, run_id: str) -> GEPASession:
    proxy.register(run_id, budget)
    lm = ProxyLM(
        proxy.base_url, run_id, role="agent",
        model=os.environ.get("LLM_MODEL") or "proxy-default",
        max_tokens=cfg.gepa.reflection_max_tokens or cfg.max_completion_tokens,
        temperature=cfg.gepa.temperature,
    )
    logger = RunLogger(out / "history.jsonl", run_id)
    monitor = FatigueMonitor(cfg.fatigue, best_known=evaluator.best_known, total_budget=budget.total_tokens)
    return GEPASession(cfg, evaluator, budget, lm, logger, out, monitor=monitor)


def write_outputs(session: GEPASession, *, arm: str, run_id: str, t0: float, extra: dict | None = None) -> dict:
    cfg, ev, budget, out = session.cfg, session.evaluator, session.budget, session.out
    best = session.best
    if best.code is not None:
        (out / "best_program.py").write_text(best.code, encoding="utf-8")
    has_best = best.program_id is not None and math.isfinite(best.score)
    env_model = os.environ.get("LLM_MODEL")
    summary = {
        "run_id": run_id,
        "problem_id": ev.problem_id,
        "problem": ev.name,
        "sleep_policy": arm,  # analysis groups runs by this key
        "arm": arm,
        "seed": cfg.seed,
        "steps_completed": session.steps_done,
        "stop_reason": stop_reason(session),
        "best_program_id": best.program_id if has_best else None,
        "best_score": best.score if has_best else None,
        "best_known": ev.best_known,
        "best_normalized": normalized_score(best.score, ev.best_known) if has_best else None,
        "best_step": best.step if has_best else None,
        "budget": budget.snapshot(),
        "llm_model": env_model,
        "config_model": cfg.model,
        "wallclock_s": round(time.monotonic() - t0, 3),
        "gepa": {
            "version": gepa_version(),
            "max_metric_calls": cfg.metric_calls,
            "sandbox_evals": session.next_step,
            "internal_metric_calls": session.gepa_internal_calls,
            "iterations": session.gepa_iterations,
            "llm_calls": session.lm.calls,
            "over_limit_calls": session.over_limit_calls,
            "error": session.error,
        },
        "config": cfg.model_dump(),
    }
    mon = session.monitor
    if mon is not None:
        summary["fatigue"] = {
            "enabled": cfg.fatigue.enabled,
            "sleeps": {d.label: sum(1 for _, x in mon.sleeps if x == d)
                       for d in (SleepDepth.FORGET, SleepDepth.NREM, SleepDepth.NREM_REM)},
            "deferrals": len(mon.deferrals),
            "final_level": mon.state.level.label,
            "final_fatigue": round(mon.state.fatigue, 6),
        }
    if extra:
        summary.update(extra)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_gepa(cfg: GEPAArmConfig, evaluator: Evaluator, budget: Budget, proxy: LLMProxy,
             out_dir: str | Path | None = None) -> dict:
    """One GEPA run: a single optimize_anything call with max_metric_calls sandbox evaluations."""
    run_id = cfg.resolved_run_id()
    out = Path(out_dir or cfg.out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    env_model = os.environ.get("LLM_MODEL")
    if cfg.model and env_model and cfg.model != env_model:
        warnings.warn(f"config model {cfg.model!r} != LLM_MODEL {env_model!r}; calls use LLM_MODEL")
    session = open_session(cfg, evaluator, budget, proxy, out, run_id)
    try:
        session.optimize(evaluator.seed_program(), cfg.gepa.background, new_evals=cfg.metric_calls)
    finally:
        proxy.unregister(run_id)
    return write_outputs(session, arm=ARM_NAMES["gepa"], run_id=run_id, t0=t0)
