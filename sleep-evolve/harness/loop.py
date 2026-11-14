"""Minimal evolution loop: sample parent -> ask LLM -> evaluate -> log -> maybe sleep.

The Budget is read from the LLMClient (llm.budget), so agent and sleep calls are always
charged to the same total. The loop stops at the first BudgetExhausted or after max_steps.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
import warnings
from pathlib import Path

from evaluators.base import INVALID_SCORE, EvalResult, Evaluator, invalid, normalized_score
from harness.budget import BudgetExhausted
from harness.config import RunConfig
from harness.logging import RunLogger
from harness.program_db import Program, ProgramDB
from llm.client import LLMClient
from sleep.base import SleepPolicy, render_steps, step_events

AGENT_SYSTEM = (
    "You are an expert research engineer improving a program in an evolutionary search. "
    "Each program defines construct(), which returns a candidate solution that an "
    "automatic evaluator scores (higher is better). Propose ONE improved program. Reply "
    "with a brief rationale, then the complete program in a single ```python code block."
)

_CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


def extract_program(text: str) -> str | None:
    """Return the last fenced code block, preferring ```python blocks."""
    py = re.findall(r"```(?:python|py)[ \t]*\n(.*?)```", text, re.DOTALL)
    blocks = py or _CODE_BLOCK.findall(text)
    return blocks[-1].strip() + "\n" if blocks else None


def build_agent_messages(task: str, block: str, recent: list[dict], parent: Program) -> list[dict]:
    parts = [f"## Task\n{task}"]
    if block:
        parts.append(f"## Consolidated notes from earlier in this run\n{block}")
    if recent:
        parts.append(f"## Most recent steps\n{render_steps(recent)}")
    score = f"{parent.score:.6g}" if parent.valid else "invalid"
    parts.append(f"## Parent program ({parent.id}, score {score})\n```python\n{parent.code}```")
    parts.append("Write an improved version of the parent program.")
    return [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]


def _safe_evaluate(evaluator: Evaluator, code: str) -> EvalResult:
    try:
        return evaluator.evaluate(code)
    except Exception as e:  # evaluators should not raise, but the loop must never crash on one
        return invalid(f"evaluator raised {type(e).__name__}: {e}")


def run(
    cfg: RunConfig,
    evaluator: Evaluator,
    llm: LLMClient,
    policy: SleepPolicy,
    out_dir: str | Path | None = None,
) -> dict:
    budget = llm.budget
    run_id = cfg.resolved_run_id()
    out = Path(out_dir or cfg.out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(out / "history.jsonl", run_id)
    rng = random.Random(cfg.seed)
    db = ProgramDB()
    history: list[dict] = []  # in-memory mirror of history.jsonl; only sleep policies see all of it
    block = ""
    t0 = time.monotonic()

    env_model = os.environ.get("LLM_MODEL")
    if cfg.model and env_model and cfg.model != env_model:
        warnings.warn(f"config model {cfg.model!r} != LLM_MODEL {env_model!r}; calls use LLM_MODEL")

    def record(step: int, prog: Program, result: EvalResult) -> None:
        prev_best = db.best()
        db.add(prog)
        delta = result.score - prev_best.score if (result.valid and prev_best) else None
        history.append(
            logger.log(
                step=step, parent_id=prog.parent_id, program_id=prog.id, program=prog.code,
                score=result.score, valid=result.valid, error=result.error, delta_vs_best=delta,
                agent_tokens=budget.agent_tokens, sleep_tokens=budget.sleep_tokens,
                event="step", details=result.details,
            )
        )

    # Step 0: the seed program (no LLM call).
    seed_code = evaluator.seed_program()
    seed_result = _safe_evaluate(evaluator, seed_code)
    record(0, Program("p0", seed_code, seed_result.score if seed_result.valid else INVALID_SCORE,
                      seed_result.valid, None, 0), seed_result)

    stop_reason = "max_steps"
    steps_done = 0
    for step in range(1, cfg.max_steps + 1):
        parent = db.sample_parent(rng)
        recent = step_events(history)[-cfg.k_recent:] if cfg.k_recent else []
        messages = build_agent_messages(evaluator.description, block, recent, parent)
        try:
            resp = llm.complete(messages, role="agent", max_tokens=cfg.max_completion_tokens)
        except BudgetExhausted:
            stop_reason = "budget_exhausted"
            break
        code = extract_program(resp.text)
        result = _safe_evaluate(evaluator, code) if code else invalid("no ```python code block in response")
        record(step, Program(f"p{step}", code or "", result.score if result.valid else INVALID_SCORE,
                             result.valid, parent.id, step), result)
        steps_done = step

        if policy.should_sleep(step, history):
            try:
                block = policy.sleep(evaluator.description, block, history, llm, budget)
            except BudgetExhausted:
                stop_reason = "budget_exhausted"
                break
            history.append(
                logger.log(
                    step=step, agent_tokens=budget.agent_tokens, sleep_tokens=budget.sleep_tokens,
                    event="sleep", details={"policy": policy.name, "block": block},
                )
            )
        if budget.exhausted:
            stop_reason = "budget_exhausted"
            break

    best = db.best()
    if best:
        (out / "best_program.py").write_text(best.code, encoding="utf-8")
    summary = {
        "run_id": run_id,
        "problem_id": evaluator.problem_id,
        "problem": evaluator.name,
        "sleep_policy": policy.name,
        "seed": cfg.seed,
        "steps_completed": steps_done,
        "stop_reason": stop_reason,
        "best_program_id": best.id if best else None,
        "best_score": best.score if best else None,
        "best_known": evaluator.best_known,
        "best_normalized": normalized_score(best.score, evaluator.best_known) if best else None,
        "best_step": best.step if best else None,
        "budget": budget.snapshot(),
        "llm_model": env_model,
        "config_model": cfg.model,
        "wallclock_s": round(time.monotonic() - t0, 3),
        "config": cfg.model_dump(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
