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

Every line also carries the CUMULATIVE evaluator-call counts `evaluations` (all roles) and
`sleep_evaluations` (harness/evals.py); summary.json["evaluations"] has the final per-role totals.
A policy with `extra_evals_after(step)` (NoSleep's eval-matched control) gets that many perturbation
evaluations of the step's parent after the step (event "extra_eval"; improving ones enter the ProgramDB).

Every line carries the LLM usage since the previous line (harness/logging.py USAGE_FIELDS).
summary.json also carries `usage` (Budget.usage_breakdown(): per role and total, with cached and
reasoning tokens), `llm_params` (the resolved request parameters, llm/chat_params.py) and `cost`
(harness/cost.py: dollars from the logged usage and the config's price table).

A BillingError (quota / spend limit, never retried) ends the run with stop_reason "billing_error":
outputs are written, then the error is re-raised so a matrix stops instead of launching more runs.

summary.json carries `metrics` (harness/metrics.py: normalized score at budget checkpoints, budget to
reach 95%/99%, sleep counts and tokens, behavior before/after sleeps, fatigue curve, memory
diagnostics, edit kinds), `arm` (RunConfig.baseline_arm, else the policy name) and `edit_mode`.

Edit formats (RunConfig.edit_mode, harness/edits.py; the same in every arm): "full" = the complete
program in a fenced block (the original behavior, prompt unchanged); "diff" = SEARCH/REPLACE blocks
against the parent shown; "auto" = either. The logged `program` is always the full resulting program.
Parent-truncation policy: build_context truncates the parent only as the LAST resort (after recent
steps, side_info and memory are gone). A diff needs the exact parent, so when the parent had to be
truncated in diff/auto mode, that step is rebuilt with the (shorter) "full" system prompt plus a note
that the parent did not fit and a complete program is required, and a SEARCH/REPLACE reply is rejected
(invalid step, `diff_error`). The step logs edit_mode "full"; parent_truncated is whether the parent
actually shown was truncated (the rebuild may just fit). The context cap still holds (both builds go
through build_context).
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
from harness.edits import (  # noqa: F401 (extract_program re-exported)
    decide_edit, extract_program, first_marker_offset, parse_full_rewrite_reason,
)
from harness.fatigue import FatigueMonitor, SleepDepth
from harness.logging import RunLogger, UsageCursor
from harness.metrics import compute_run_metrics
from harness.perturb import perturb_program
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

_EDIT_FORMAT = (
    "<<<<<<< SEARCH\n"
    "(lines copied EXACTLY from the parent program, same indentation)\n"
    "=======\n"
    "(the new lines that replace them)\n"
    ">>>>>>> REPLACE\n"
)
_EDIT_EXAMPLE = (
    "Example diff:\n"
    "<<<<<<< SEARCH\n"
    "    n_iters = 100\n"
    "=======\n"
    "    n_iters = 400\n"
    ">>>>>>> REPLACE\n"
)
_EDIT_RULES = (
    "Each SEARCH section must copy whole lines exactly, character for character, from the parent program "
    "shown, and must match exactly one place in it. Blocks are applied in order; you may use several. "
    "An empty REPLACE section deletes the lines."
)

AGENT_SYSTEM_AUTO = (
    "You are an expert research engineer improving a program in an evolutionary search. "
    "Each program defines construct(), which returns a candidate solution that an "
    "automatic evaluator scores (higher is better). Propose ONE improved program. Reply "
    "with a brief rationale, then your edit. DIFFS ARE THE DEFAULT: reply with one or more "
    "SEARCH/REPLACE blocks that edit the parent program:\n"
    + _EDIT_FORMAT + _EDIT_RULES + "\n" + _EDIT_EXAMPLE +
    "Write the complete program in a single ```python code block (a full rewrite) ONLY when a diff is "
    "not feasible: the parent program is not shown or was truncated, the change restructures most of "
    "the program, or your previous diff attempt failed to apply. A full rewrite must be preceded by one "
    "line:\nFULL_REWRITE_REASON: <why a diff was not feasible>\n"
    "Do not mix the two formats in one reply."
)

AGENT_SYSTEM_DIFF = (
    "You are an expert research engineer improving a program in an evolutionary search. "
    "Each program defines construct(), which returns a candidate solution that an "
    "automatic evaluator scores (higher is better). Propose ONE improved program. Reply "
    "with a brief rationale, then ONLY SEARCH/REPLACE blocks that edit the parent program:\n"
    + _EDIT_FORMAT + _EDIT_RULES + " A complete program without SEARCH/REPLACE blocks is rejected.\n"
    + _EDIT_EXAMPLE
)

AGENT_SYSTEMS = {"full": AGENT_SYSTEM, "auto": AGENT_SYSTEM_AUTO, "diff": AGENT_SYSTEM_DIFF}

TRUNCATED_PARENT_INSTRUCTION = (
    "Write an improved version of the parent program. The parent did not fit the context in full, "
    "so SEARCH/REPLACE diffs are not accepted this step: reply with the complete program in a "
    "single ```python code block."
)

AWAKE_HEADER_INSTRUCTIONS = (
    "\n\nBefore the code block, write these four header lines exactly:\n"
    "APPROACH_TAGS: 2-5 short lowercase tags naming your approach, comma-separated "
    "(e.g. hex-lattice, lbfgs-refine)\n"
    "EXPECTED_SCORE: your numeric prediction of the score this program will get\n"
    "RATIONALE: one sentence on why it should work\n"
    "RELEVANT_MEMORY: ids of memory items (like m3) that informed it, or none"
)
AWAKE_HEADER_INSTRUCTIONS_EDITS = AWAKE_HEADER_INSTRUCTIONS.replace(
    "Before the code block,", "Before the code block or the first SEARCH/REPLACE block,")

SLEEP_TOP_LEVEL = ("pressure", "sleep_mode", "items_added", "items_evicted", "replays_wasted")

_HEADER = re.compile(
    r"^[ \t>*_#-]*(APPROACH[ _]TAGS|EXPECTED[ _]SCORE|RATIONALE|RELEVANT[ _]MEMORY)[*_ \t]*[:=][ \t]*(.*)$",
    re.IGNORECASE | re.MULTILINE,
)
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def parse_awake_header(text: str) -> dict:
    """Lenient parse of APPROACH_TAGS / EXPECTED_SCORE / RATIONALE / RELEVANT_MEMORY lines.

    Only the text before the first code fence (or the first SEARCH marker line of a diff) is
    searched; the first occurrence of each key wins. Missing or garbled lines give None.
    """
    out: dict = {"approach_tags": None, "expected_score": None, "rationale": None, "relevant_memory": None}
    head = (text or "").split("```", 1)[0]
    cut = first_marker_offset(head)
    if cut is not None:
        head = head[:cut]
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


def agent_system(edit_mode: str = "full", awake_tags: bool = False) -> str:
    """The agent system prompt for an edit mode ("full" is the original prompt, byte for byte)."""
    if edit_mode == "full":
        return AGENT_SYSTEM + (AWAKE_HEADER_INSTRUCTIONS if awake_tags else "")
    return AGENT_SYSTEMS[edit_mode] + (AWAKE_HEADER_INSTRUCTIONS_EDITS if awake_tags else "")


def agent_context(task: str, block: str, recent: list[dict], parent: Program, *, cap: int | None = None,
                  awake_tags: bool = False, parent_side_info: dict | None = None,
                  edit_mode: str = "full") -> tuple[list[dict], dict]:
    """(messages, stats) for one agent call; bounded by `cap` (see harness/context.py).

    stats["edit_mode"] is the mode in force for the step: `edit_mode`, or "full" when diffs were
    allowed but the parent had to be truncated (the context is then rebuilt with the full-rewrite
    prompt and TRUNCATED_PARENT_INSTRUCTION, and stats["diffs_forbidden"] is True; see the module
    docstring)."""
    kw = dict(task=task, memory=block, recent=recent, parent_id=parent.id,
              parent_score=parent.score if parent.valid else None, parent_code=parent.code,
              parent_side_info=parent_side_info, cap=cap)
    msgs, stats = build_context(system=agent_system(edit_mode, awake_tags), **kw)
    stats["edit_mode"], stats["diffs_forbidden"] = edit_mode, False
    if edit_mode != "full" and stats["program_truncated"]:
        msgs, stats = build_context(system=agent_system("full", awake_tags),
                                    instruction=TRUNCATED_PARENT_INSTRUCTION, **kw)
        stats["edit_mode"], stats["diffs_forbidden"] = "full", True
    return msgs, stats



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
    env_sleep_model = os.environ.get("LLM_SLEEP_MODEL")
    if cfg.sleep_model and env_sleep_model and cfg.sleep_model != env_sleep_model:
        warnings.warn(f"config sleep_model {cfg.sleep_model!r} != LLM_SLEEP_MODEL "
                      f"{env_sleep_model!r}; sleep calls use LLM_SLEEP_MODEL")
    sleep_model = env_sleep_model or cfg.sleep_model
    sleep_llm = llm.with_model(sleep_model) if sleep_model else llm

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
    # eval-matched no_sleep control (sleep/no_sleep.py extra_eval_schedule): perturbation evals
    extra_evals_after = getattr(policy, "extra_evals_after", None)
    extra_evaluator = evaluator.tagged("extra")
    extra_rng = random.Random(f"extra_evals|{cfg.seed}")  # separate stream: parent sampling is unchanged
    extra_sigma = float(getattr(policy, "extra_eval_sigma", 0.05))

    def eval_fields() -> dict:
        return {"evaluations": evals.used, "sleep_evaluations": evals.sleep_evals}

    def run_extra_evals(step: int, parent: Program, k: int) -> None:
        """k perturbations of `parent`, each one evaluator call (role "extra"). An improving one
        (valid and better than the parent) enters the ProgramDB, so the control can build on it.
        Logged as event "extra_eval", never as a step. EvalExhausted propagates."""
        for i in range(k):
            code, n_changed = perturb_program(parent.code, extra_rng, extra_sigma)
            result = _safe_evaluate(extra_evaluator, code)
            prev_best = db.best()
            entered = bool(result.valid and (not parent.valid or result.score > parent.score))
            pid = f"p{step}x{i}"
            if entered:
                db.add(Program(pid, code, result.score, True, parent.id, step))
                side_infos[pid] = result.side_info
            history.append(logger.log(
                step=step, parent_id=parent.id, program_id=pid, program=code, score=result.score,
                valid=result.valid, error=result.error, event="extra_eval",
                delta_vs_best=result.score - prev_best.score if (result.valid and prev_best) else None,
                details={"entered_db": entered, "perturbed_literals": n_changed}, side_info=result.side_info,
                agent_tokens=budget.agent_tokens, sleep_tokens=budget.sleep_tokens, **eval_fields(),
                **usage_cursor.take(),
            ))

    def record(step: int, prog: Program, result: EvalResult, awake: dict | None = None,
               context_tokens: int | None = None, edit_fields: dict | None = None) -> dict:
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
            context_tokens=context_tokens, **awake, **(edit_fields or {}),
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
                           **eval_fields(), **usage_cursor.take())
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
                                      awake_tags=awake_tags, parent_side_info=side_infos.get(parent.id),
                                      edit_mode=cfg.edit_mode)
        try:
            resp = llm.complete(messages, role="agent", max_tokens=cfg.max_completion_tokens)
        except BudgetExhausted:
            stop_reason = "budget_exhausted"
            break
        except BillingError as e:
            stop_reason, billing_error = "billing_error", e
            break
        edit = decide_edit(resp.text, parent.code, cfg.edit_mode, diffs_forbidden=ctx["diffs_forbidden"])
        code = edit.code  # always the FULL program (a diff is already applied to the parent)
        awake = parse_awake_header(resp.text) if awake_tags else None
        if awake and awake.get("relevant_memory"):
            # citations are self-reported: flag ids the agent cannot have seen in its memory block
            awake["relevant_memory_unknown"] = [m for m in awake["relevant_memory"]
                                                if f"[{m}]" not in block] or None
        try:
            result = _safe_evaluate(evaluator, code) if code else invalid(edit.error)
        except EvalExhausted:
            stop_reason = "eval_limit"
            break
        edit_fields = {
            "edit_mode": ctx["edit_mode"], "edit_kind": edit.kind, "diff_blocks": edit.stats.get("blocks"),
            "diff_lines_changed": edit.stats.get("lines_changed"), "diff_fuzzy_blocks": edit.stats.get("fuzzy_blocks"),
            "diff_error": edit.diff_error, "parent_truncated": bool(ctx["program_truncated"]),
            "full_rewrite_reason": parse_full_rewrite_reason(resp.text) if edit.kind == "full" else None,
        }
        copy = edit.stats.get("copy") or {}
        edit_fields.update({
            "search_blocks": copy.get("blocks"), "search_blocks_exact": copy.get("exact"),
            "search_blocks_near": copy.get("near"), "search_copy_score": copy.get("copy_score"),
        })
        rec = record(step, Program(f"p{step}", code or "", result.score if result.valid else INVALID_SCORE,
                                   result.valid, parent.id, step), result, awake, ctx["tokens"], edit_fields)
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
                block = policy.sleep(evaluator.description, block, history, sleep_llm, budget)
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
                    **eval_fields(), **usage_cursor.take(),
                )
            )
        n_extra = extra_evals_after(step) if extra_evals_after is not None else 0
        if n_extra:
            try:
                run_extra_evals(step, parent, n_extra)
            except EvalExhausted:
                stop_reason = "eval_limit"
                break
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
        "edit_mode": cfg.edit_mode,
        "steps_completed": steps_done,
        "stop_reason": stop_reason,
        "best_program_id": best.id if best else None,
        "best_score": best.score if best else None,
        "best_known": evaluator.best_known,
        "best_normalized": normalized_score(best.score, evaluator.best_known) if best else None,
        "best_step": best.step if best else None,
        "budget": budget.snapshot(),
        "evaluations": {**evals.snapshot(), "extra_scheduled": sum(getattr(policy, "extra_eval_schedule", {})
                                                                   .values())},
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
        "sleep_model": sleep_model,
        "wallclock_s": round(time.monotonic() - t0, 3),
        "error": str(billing_error) if billing_error is not None else None,
        "config": cfg.model_dump(),
    }
    summary.update(usage_summary(llm, budget, cfg, sleep_llm))
    try:  # summary metrics (harness/metrics.py); a bug there must not lose the run's outputs
        summary["metrics"] = compute_run_metrics(
            history, best_known=evaluator.best_known, total_budget=budget.total_tokens, budget=budget.snapshot(),
            scheduled=getattr(policy, "scheduled_sleeps", None), cfg=cfg.metrics, fatigue_cfg=cfg.fatigue,
            arm=summary["arm"], sleep_policy=policy.name, evaluations=summary["evaluations"],
            edit_mode=cfg.edit_mode)
    except Exception as e:  # pragma: no cover - defensive
        summary["metrics"] = {"error": f"{type(e).__name__}: {e}"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if billing_error is not None:
        raise billing_error
    return summary


def usage_summary(llm: object, budget: object, cfg: RunConfig, sleep_llm: object | None = None) -> dict:
    """summary.json `usage`, `llm_params` and `cost` (shared with the GEPA arms).

    The priced model is the one actually called (the transport's model, i.e. LLM_MODEL), else the
    config's `model` (mock runs). When `sleep_llm` is a different model (RunConfig.sleep_model /
    LLM_SLEEP_MODEL), the `sleep` role is priced at that model and the agent role at the agent's."""
    params = getattr(llm, "chat_params", None)
    model = getattr(llm, "model", None) or cfg.model
    sleep_model = getattr(sleep_llm, "model", None) if sleep_llm is not None else None
    usage = budget.usage_breakdown()
    table = price_table(cfg.prices)
    cost = cost_block(usage, model, table)
    if sleep_model and sleep_model != model:
        scost = cost_block(usage, sleep_model, table)
        a, s = cost["by_role"]["agent"], scost["by_role"]["sleep"]
        cost["sleep_model"] = sleep_model
        cost["sleep_price_per_1m"] = scost["price_per_1m"]
        cost["by_role"]["sleep"] = s
        cost["total"] = round(a + s, 6) if a is not None and s is not None else None
        cost["complete"] = cost["total"] is not None
        cost["notes"] += [f"sleep role priced at {sleep_model!r}"] + scost["notes"]
    sparams = getattr(sleep_llm, "chat_params", None)
    return {"usage": usage, "llm_params": params.to_dict() if params is not None else None,
            "sleep_llm_params": sparams.to_dict() if sparams is not None else None,
            "cost": cost}
