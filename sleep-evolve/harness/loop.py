"""Minimal evolution loop: sample parent -> ask LLM -> evaluate -> log -> maybe sleep.

The Budget is read from the LLMClient (llm.budget), so agent and sleep calls are always
charged to the same total. The loop stops at the first BudgetExhausted or after max_steps.

The agent prompt is built by harness.context.build_context and never exceeds the context
cap (min of RunConfig.context_cap_tokens and the policy's optional context_cap_tokens).

Optional policy hooks (called only if the policy defines them; NoSleep/Summarize don't):
  bind(evaluator=...)          once, before step 0 (FullSleep uses it for replays)
  observe(step, record) -> {"pressure", "perseveration_hit", "perseveration_items"}
                               before each step is logged; values are logged with the step
  last_metrics                 dict read after sleep(); known keys become top-level log
                               fields, the rest go into the sleep event's details
  awake_tags, context_cap_tokens   attributes (see RunConfig for the run-level overrides)
  set_fatigue(reading)         before each should_sleep: read-only FatigueReading (level,
                               recommended depth, stagnant, remaining budget)
  last_deferred                bool set by should_sleep: a sleep was due but deferred
  last_depth                   depth the policy actually ran in sleep() ("forget" | "nrem" |
                               "nrem_rem" | "none"); missing -> fatigue.offline_default_depth
  select_recent(recent, step)  filters/slims the last-K step records shown to the agent

Fatigue (harness/fatigue.py) is measured here for EVERY arm, no_sleep included: each step
record is fed to the FatigueMonitor, the policy decides, the loop tells the monitor what
happened (on_sleep / on_deferred), and only then is the step line written, so it carries
the observed fatigue plus sleep_triggered / sleep_deferred / sleep_depth. Line order in the
JSONL is unchanged (step line, then its sleep line).

Every line carries the LLM usage since the previous line (harness/logging.py USAGE_FIELDS).
summary.json also carries `usage` (Budget.usage_breakdown(): per role and total, with cached and
reasoning tokens), `llm_params` (the resolved request parameters, llm/chat_params.py) and `cost`
(harness/cost.py: dollars from the logged usage and the config's price table).

A BillingError (quota / spend limit, never retried) ends the run with stop_reason "billing_error":
outputs are written, then the error is re-raised so a matrix stops instead of launching more runs.

summary.json carries `metrics` (harness/metrics.py: normalized score at budget checkpoints, budget to
reach 95%/99%, sleep counts and tokens, behavior before/after sleeps, fatigue curve, memory
diagnostics) and `arm` (RunConfig.baseline_arm, else the policy name).
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
import warnings
from pathlib import Path

from evaluators.base import INVALID_SCORE, EvalResult, Evaluator, invalid, normalized_score
from harness.budget import BudgetExhausted
from harness.config import RunConfig
from harness.context import build_context, count_tokens
from harness.evals import CountingEvaluator, EvalBudget, EvalExhausted
from harness.cost import cost_block, price_table
from harness.fatigue import FatigueMonitor, SleepDepth
from harness.logging import RunLogger, UsageCursor
from harness.metrics import compute_run_metrics
from harness.program_db import Program, ProgramDB
from llm.chat_params import BillingError
from llm.client import LLMClient
from sleep.base import SleepPolicy, step_events
from sleep.memory import normalize_tags
from sleep.pressure import compute_surprise

AGENT_SYSTEM = (
    "You are an expert research engineer improving a program in an evolutionary search. "
    "Each program defines construct(), which returns a candidate solution that an "
    "automatic evaluator scores (higher is better). Propose ONE improved program. Reply "
    "with a brief rationale, then the complete program in a single ```python code block."
)

AWAKE_HEADER_INSTRUCTIONS = (
    "\n\nBefore the code block, write these four header lines exactly:\n"
    "APPROACH_TAGS: 2-5 short lowercase tags naming your approach, comma-separated "
    "(e.g. hex-lattice, lbfgs-refine)\n"
    "EXPECTED_SCORE: your numeric prediction of the score this program will get\n"
    "RATIONALE: one sentence on why it should work\n"
    "RELEVANT_MEMORY: ids of memory items (like m3) that informed it, or none"
)

SLEEP_TOP_LEVEL = ("pressure", "sleep_mode", "items_added", "items_evicted", "replays_wasted")

_CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)
_HEADER = re.compile(
    r"^[ \t>*_#-]*(APPROACH[ _]TAGS|EXPECTED[ _]SCORE|RATIONALE|RELEVANT[ _]MEMORY)[*_ \t]*[:=][ \t]*(.*)$",
    re.IGNORECASE | re.MULTILINE,
)
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def extract_program(text: str) -> str | None:
    """Return the last fenced code block, preferring ```python blocks."""
    py = re.findall(r"```(?:python|py)[ \t]*\n(.*?)```", text, re.DOTALL)
    blocks = py or _CODE_BLOCK.findall(text)
    return blocks[-1].strip() + "\n" if blocks else None


def parse_awake_header(text: str) -> dict:
    """Lenient parse of APPROACH_TAGS / EXPECTED_SCORE / RATIONALE / RELEVANT_MEMORY lines.

    Only the text before the first code fence is searched; the first occurrence of each key
    wins. Missing or garbled lines give None.
    """
    out: dict = {"approach_tags": None, "expected_score": None, "rationale": None, "relevant_memory": None}
    head = (text or "").split("```", 1)[0]
    for m in _HEADER.finditer(head):
        key = m.group(1).upper().replace(" ", "_")
        val = m.group(2).strip().strip("*_`").strip()
        if key == "APPROACH_TAGS" and out["approach_tags"] is None:
            out["approach_tags"] = normalize_tags(re.sub(r"[\[\](){}\"']", "", val)) or None
        elif key == "EXPECTED_SCORE" and out["expected_score"] is None:
            num = _NUMBER.search(val)
            if num:
                x = float(num.group(0))
                out["expected_score"] = x if math.isfinite(x) else None
        elif key == "RATIONALE" and out["rationale"] is None:
            out["rationale"] = val[:500] or None
        elif key == "RELEVANT_MEMORY" and out["relevant_memory"] is None:
            ids = list(dict.fromkeys(re.findall(r"\b([a-z]\d+)\b", val.lower())))
            out["relevant_memory"] = ids or None
    return out


def agent_context(task: str, block: str, recent: list[dict], parent: Program, *, cap: int | None = None,
                  awake_tags: bool = False, parent_side_info: dict | None = None) -> tuple[list[dict], dict]:
    """(messages, stats) for one agent call; bounded by `cap` (see harness/context.py)."""
    system = AGENT_SYSTEM + (AWAKE_HEADER_INSTRUCTIONS if awake_tags else "")
    return build_context(
        system=system, task=task, memory=block, recent=recent, parent_id=parent.id,
        parent_score=parent.score if parent.valid else None, parent_code=parent.code,
        parent_side_info=parent_side_info, cap=cap,
    )



def _safe_evaluate(evaluator: Evaluator, code: str) -> EvalResult:
    try:
        return evaluator.evaluate(code)
    except EvalExhausted:
        raise  # the shared eval cap stops the run; it is not a program failure
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
    evals = EvalBudget(cfg.max_evaluations)
    evaluator = CountingEvaluator(evaluator, evals)
    run_id = cfg.resolved_run_id()
    out = Path(out_dir or cfg.out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(out / "history.jsonl", run_id)
    usage_cursor = UsageCursor(budget)
    billing_error: BillingError | None = None
    rng = random.Random(cfg.seed)
    db = ProgramDB()
    history: list[dict] = []  # in-memory mirror of history.jsonl; only sleep policies see all of it
    block = ""
    t0 = time.monotonic()

    env_model = os.environ.get("LLM_MODEL")
    if cfg.model and env_model and cfg.model != env_model:
        warnings.warn(f"config model {cfg.model!r} != LLM_MODEL {env_model!r}; calls use LLM_MODEL")

    awake_tags = cfg.awake_tags if cfg.awake_tags is not None else bool(getattr(policy, "awake_tags", False))
    caps = [c for c in (cfg.context_cap_tokens, getattr(policy, "context_cap_tokens", None)) if c]
    context_cap = min(caps) if caps else None
    observe = getattr(policy, "observe", None)
    if callable(getattr(policy, "bind", None)):
        policy.bind(evaluator=evaluator.tagged("sleep"))  # replay evals count separately
    side_infos: dict[str, dict] = {}  # program id -> side_info, shown for the sampled parent
    monitor = FatigueMonitor(cfg.fatigue, best_known=evaluator.best_known, total_budget=budget.total_tokens)
    set_fatigue = getattr(policy, "set_fatigue", None) if cfg.fatigue.enabled else None
    select_recent = getattr(policy, "select_recent", None)
    default_depth = SleepDepth.parse(cfg.fatigue.offline_default_depth)

    def record(step: int, prog: Program, result: EvalResult, awake: dict | None = None,
               context_tokens: int | None = None) -> dict:
        """Build the step record, feed it to the policy observe hook and the fatigue monitor,
        and append it to `history` (not yet written: see commit_step)."""
        prev_best = db.best()
        db.add(prog)
        side_infos[prog.id] = result.side_info
        delta = result.score - prev_best.score if (result.valid and prev_best) else None
        awake = awake or {}
        fields = dict(
            step=step, parent_id=prog.parent_id, program_id=prog.id, program=prog.code,
            score=result.score, valid=result.valid, error=result.error, delta_vs_best=delta,
            event="step", details=dict(result.details), side_info=result.side_info,
            context_tokens=context_tokens, **awake,
            surprise=compute_surprise(result.score, result.valid, awake.get("expected_score"),
                                      evaluator.best_known),
        )
        if observe is not None:
            obs = observe(step, dict(fields, score=result.score if result.valid else None)) or {}
            fields["pressure"] = obs.get("pressure")
            fields["perseveration_hit"] = obs.get("perseveration_hit")
            if obs.get("perseveration_items"):
                fields["details"]["perseveration_items"] = obs["perseveration_items"]
        rec = logger.build(agent_tokens=budget.agent_tokens, sleep_tokens=budget.sleep_tokens, **fields,
                           **usage_cursor.take())
        monitor.observe(rec)
        rec.update(monitor.log_fields())
        history.append(rec)
        return rec

    def commit_step(rec: dict) -> None:
        """Write the step line with the observed fatigue and the sleep decision taken after it."""
        rec.update(monitor.log_fields())
        logger.write(rec)

    # Step 0: the seed program (no LLM call).
    seed_code = evaluator.seed_program()
    seed_result = _safe_evaluate(evaluator, seed_code)
    commit_step(record(0, Program("p0", seed_code, seed_result.score if seed_result.valid else INVALID_SCORE,
                                  seed_result.valid, None, 0), seed_result))

    stop_reason = "max_steps"
    steps_done = 0
    for step in range(1, cfg.max_steps + 1):
        parent = db.sample_parent(rng)
        recent = step_events(history)[-cfg.k_recent:] if cfg.k_recent else []
        if select_recent is not None:
            recent = select_recent(recent, step)
        messages, ctx = agent_context(evaluator.description, block, recent, parent, cap=context_cap,
                                      awake_tags=awake_tags, parent_side_info=side_infos.get(parent.id))
        try:
            resp = llm.complete(messages, role="agent", max_tokens=cfg.max_completion_tokens)
        except BudgetExhausted:
            stop_reason = "budget_exhausted"
            break
        except BillingError as e:
            stop_reason, billing_error = "billing_error", e
            break
        code = extract_program(resp.text)
        awake = parse_awake_header(resp.text) if awake_tags else None
        try:
            result = _safe_evaluate(evaluator, code) if code else invalid(
                "no ```python code block in response")
        except EvalExhausted:
            stop_reason = "eval_limit"
            break
        rec = record(step, Program(f"p{step}", code or "", result.score if result.valid else INVALID_SCORE,
                                   result.valid, parent.id, step), result, awake, ctx["tokens"])
        steps_done = step

        if set_fatigue is not None:
            set_fatigue(monitor.reading(budget_remaining=budget.remaining))
        for attr, empty in (("last_deferred", False), ("last_depth", None)):
            if hasattr(policy, attr):
                setattr(policy, attr, empty)
        wants_sleep = policy.should_sleep(step, history)
        if getattr(policy, "last_deferred", False):
            monitor.on_deferred()
        if not wants_sleep:
            commit_step(rec)
        else:
            if hasattr(policy, "last_metrics"):
                policy.last_metrics = {}
            try:
                block = policy.sleep(evaluator.description, block, history, llm, budget)
            except BudgetExhausted:
                commit_step(rec)
                stop_reason = "budget_exhausted"
                break
            except BillingError as e:
                commit_step(rec)
                stop_reason, billing_error = "billing_error", e
                break
            except EvalExhausted:
                commit_step(rec)
                stop_reason = "eval_limit"
                break
            depth = getattr(policy, "last_depth", None)
            monitor.on_sleep(default_depth if depth is None else depth)
            commit_step(rec)
            metrics = dict(getattr(policy, "last_metrics", None) or {})
            top = {k: metrics.pop(k, None) for k in SLEEP_TOP_LEVEL}
            metrics.pop("block_tokens", None)
            metrics.pop("policy", None)
            metrics.pop("sleep_depth", None)
            history.append(
                logger.log(
                    step=step, agent_tokens=budget.agent_tokens, sleep_tokens=budget.sleep_tokens,
                    event="sleep", details={"policy": policy.name, "block": block, **metrics},
                    block_tokens=count_tokens(block), **top, **monitor.log_fields(monitor.current()),
                    **usage_cursor.take(),
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
        "arm": cfg.baseline_arm or policy.name,
        "seed": cfg.seed,
        "steps_completed": steps_done,
        "stop_reason": stop_reason,
        "best_program_id": best.id if best else None,
        "best_score": best.score if best else None,
        "best_known": evaluator.best_known,
        "best_normalized": normalized_score(best.score, evaluator.best_known) if best else None,
        "best_step": best.step if best else None,
        "budget": budget.snapshot(),
        "evaluations": evals.snapshot(),
        "fatigue": {
            "enabled": cfg.fatigue.enabled,
            "sleeps": {d.label: sum(1 for _, x in monitor.sleeps if x == d)
                       for d in (SleepDepth.FORGET, SleepDepth.NREM, SleepDepth.NREM_REM)},
            "deferrals": len(monitor.deferrals),
            "final_level": monitor.state.level.label,
            "final_fatigue": round(monitor.state.fatigue, 6),
        },
        "llm_model": env_model,
        "config_model": cfg.model,
        "wallclock_s": round(time.monotonic() - t0, 3),
        "error": str(billing_error) if billing_error is not None else None,
        "config": cfg.model_dump(),
    }
    summary.update(usage_summary(llm, budget, cfg))
    try:  # summary metrics (harness/metrics.py); a bug there must not lose the run's outputs
        summary["metrics"] = compute_run_metrics(
            history, best_known=evaluator.best_known, total_budget=budget.total_tokens, budget=budget.snapshot(),
            scheduled=getattr(policy, "scheduled_sleeps", None), cfg=cfg.metrics, fatigue_cfg=cfg.fatigue,
            arm=summary["arm"], sleep_policy=policy.name)
    except Exception as e:  # pragma: no cover - defensive
        summary["metrics"] = {"error": f"{type(e).__name__}: {e}"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if billing_error is not None:
        raise billing_error
    return summary


def usage_summary(llm: object, budget: object, cfg: RunConfig) -> dict:
    """summary.json `usage`, `llm_params` and `cost` (shared with the GEPA arms).

    The priced model is the one actually called (the transport's model, i.e. LLM_MODEL), else the
    config's `model` (mock runs)."""
    params = getattr(llm, "chat_params", None)
    model = getattr(llm, "model", None) or cfg.model
    usage = budget.usage_breakdown()
    return {"usage": usage, "llm_params": params.to_dict() if params is not None else None,
            "cost": cost_block(usage, model, price_table(cfg.prices))}
