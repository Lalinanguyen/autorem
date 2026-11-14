"""GEPA + sleep arm: GEPA in rounds of k metric calls, FullSleep between rounds.

    round 0: optimize_anything(seed_program, background=B)                 -> k sandbox evals
    sleep:   FullSleep.consolidate(task, history.jsonl records, llm, budget, step=last step)
    round r: optimize_anything(best program so far, background=B + memory_section())
    ... until max_metric_calls sandbox evals (matched to max_steps) or the Budget is spent.

API basis (gepa 0.1.4, gepa/optimize_anything.py): `optimize_anything(seed_candidate, *,
evaluator, objective, background, config)`. `background` is free text placed under
"## Domain Context & Constraints" in every reflection prompt (`_build_reflection_prompt_template`),
and `seed_candidate` (str) is the starting candidate, so both mechanisms the spec asks for exist.
There is no way to change `background` inside a running optimize_anything call, hence one call per
round. Each round starts a fresh GEPA state from the best candidate (its Pareto pool and lineage
are not carried over; the memory section is the only cross-round channel besides the seed).

Budget: GEPA's reflection calls go through the proxy (role "agent"); sleep calls go through
`sleep_llm`, an `LLMClient` that must share the run's Budget (role "sleep"). Sleep probes (FullSleep
replays) use the evaluator but are not search steps: they are not logged as "step" events and do
not count toward max_metric_calls (same rule as harness/loop.py).

Fatigue: the session's FatigueMonitor observes every step (arms/gepa_arm.py). After each sleep the
monitor is told its depth (nrem, or nrem_rem for a REM pass) and the sleep line carries the post-sleep
fatigue fields. The step line before it is already written, so its sleep_triggered stays false.
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
from typing import Any, Callable

from arms.gepa_arm import ARM_NAMES, GEPAArmConfig, open_session, write_outputs
from evaluators.base import Evaluator
from harness.budget import Budget, BudgetExhausted
from harness.fatigue import FATIGUE_FIELDS, SleepDepth
from harness.logging import FIELDS
from infra.llm_proxy import LLMProxy

# Keys RunLogger sets itself or that only step events carry; never taken from sleep metrics.
_RESERVED = {
    "run_id", "step", "parent_id", "program_id", "program", "score", "valid", "error",
    "delta_vs_best", "agent_tokens", "sleep_tokens", "wallclock", "event", "details", "side_info",
}

MEMORY_HEADER = "## Memory from earlier rounds (consolidated during sleep)"


def compose_background(base: str, memory: str) -> str:
    memory = (memory or "").strip()
    return f"{base}\n\n{MEMORY_HEADER}\n{memory}" if memory else base


def make_full_sleep(params: dict) -> Any:
    from sleep.full import FullSleep

    return FullSleep(**params)


def run_gepa_sleep(
    cfg: GEPAArmConfig,
    evaluator: Evaluator,
    budget: Budget,
    proxy: LLMProxy,
    sleep_llm: Any,
    out_dir: str | Path | None = None,
    *,
    sleep_factory: Callable[[dict], Any] = make_full_sleep,
) -> dict:
    if getattr(sleep_llm, "budget", None) is not budget:
        raise ValueError("sleep_llm must be an LLMClient on the run's Budget (matched budget)")
    run_id = cfg.resolved_run_id()
    out = Path(out_dir or cfg.out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    env_model = os.environ.get("LLM_MODEL")
    if cfg.model and env_model and cfg.model != env_model:
        warnings.warn(f"config model {cfg.model!r} != LLM_MODEL {env_model!r}; calls use LLM_MODEL")

    policy = sleep_factory(dict(cfg.sleep_policies.get(cfg.gepa.sleep_policy, {})))
    if callable(getattr(policy, "bind", None)):
        policy.bind(evaluator=evaluator)
    session = open_session(cfg, evaluator, budget, proxy, out, run_id)
    observe = getattr(policy, "observe", None)
    session.observe = observe if callable(observe) else None

    task = evaluator.description
    k = cfg.gepa.round_metric_calls
    seed = evaluator.seed_program()
    background = cfg.gepa.background
    rounds = sleeps = 0
    stalled = False
    try:
        while True:
            session.current_round = rounds
            before = session.next_step
            session.optimize(seed, background, new_evals=k)
            rounds += 1
            done = session.steps_done >= cfg.metric_calls
            if done or session.error or session.budget_exhausted() or session.llm_failed():
                break
            if session.next_step == before:
                stalled = True  # gepa stopped on its internal limit without a new candidate
                break
            step = session.steps_done
            try:
                metrics = policy.consolidate(task, session.history, sleep_llm, budget, step=step) or {}
            except BudgetExhausted:
                break
            memory = policy.memory_section(cfg.gepa.memory_token_cap)
            sleeps += 1
            if session.monitor is not None:
                rem = metrics.get("sleep_mode") in ("rem", "nrem+rem")
                session.monitor.on_sleep(SleepDepth.NREM_REM if rem else SleepDepth.NREM)
            _log_sleep(session, policy, metrics, memory, step=step, round_=rounds - 1)
            if budget.exhausted:
                break
            background = compose_background(cfg.gepa.background, memory)
            seed = session.best.code or seed
    finally:
        proxy.unregister(run_id)

    extra = {"rounds": rounds, "sleeps": sleeps, "round_metric_calls": k, "stalled": stalled}
    return write_outputs(session, arm=ARM_NAMES["gepa_sleep"], run_id=run_id, t0=t0, extra=extra)


def _log_sleep(session: Any, policy: Any, metrics: dict, memory: str, *, step: int, round_: int) -> None:
    m = dict(metrics)
    m.pop("policy", None)
    m.pop("block", None)
    top = {key: m.pop(key) for key in list(m) if key in FIELDS and key not in _RESERVED}
    mon = getattr(session, "monitor", None)
    if mon is not None:
        top = {k: v for k, v in top.items() if k not in FATIGUE_FIELDS}
        top.update(mon.log_fields(mon.current()))
    details = {"policy": getattr(policy, "name", "full_sleep"), "block": memory, "round": round_, **m}
    session.history.append(
        session.logger.log(
            step=step, agent_tokens=session.budget.agent_tokens, sleep_tokens=session.budget.sleep_tokens,
            event="sleep", details=details, **top,
        )
    )
