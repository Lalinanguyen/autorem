"""Per-run summary metrics for the baseline comparison suite (summary.json["metrics"]).

Computed from the run's own JSONL records (harness/logging.py), so the same function runs online at
the end of harness/loop.py and offline over any history.jsonl (analysis/compare_baselines.py).

Budget axis: "budget used" at a step line is the cumulative agent + sleep tokens on that line
(sleep tokens count: every arm is charged on the same Budget), as a fraction of the run's total
budget. The best-so-far normalized score (evaluators.base.normalized_score, 1.0 = best_known) at
budget fraction f is the best over step lines with used <= f * total; a run that stopped before
spending f * total (max_steps) carries its final best forward, and `budget_used_fraction` says how
far it got.

Schema (METRICS_SCHEMA; every key always present, null when not computable):
  schema                    "baseline_metrics/v1"
  arm, sleep_policy         the suite arm (RunConfig.baseline_arm) and the policy name
  best_known                float
  checkpoints               {"0.1": norm | null, "0.25": ..., ...}  best-so-far normalized score
  reach                     {"0.95": {"reached": bool, "budget_fraction": float | null, "step": int | null}, ...}
                            first step whose best-so-far normalized score >= target; not reached ->
                            reached=false, budget_fraction=null (censored: analysis treats it as >= 1.0)
  final_best_normalized     float | null
  budget_used_fraction      used / total at the end
  steps_completed           int (last agent step)
  sleep                     {"llm_sleeps": n (depth nrem or nrem_rem: the count matched arms copy),
                             "forgets": n (code-only, zero LLM calls), "by_depth": {...},
                             "llm_sleep_steps": [...], "forget_steps": [...],
                             "scheduled": n | null (explicit schedules only), "realized": n (= llm_sleeps),
                             "deferrals": n}
  tokens                    {"total_budget", "used", "agent", "sleep", "sleep_share_of_budget",
                             "sleep_share_of_used", "agent_calls", "sleep_calls", "overshoot"}
  behavior                  {"window": W,
                             "overall": RATES over all agent steps,
                             "before_sleep": mean RATES over the W steps up to each LLM sleep,
                             "after_sleep": mean RATES over the W steps after each LLM sleep,
                             "around_sleeps": [{"step", "depth", "before": RATES, "after": RATES}, ...]}
                            RATES = {"perseveration_rate", "invalid_rate", "calibration_error", "n"}
  fatigue_curve             {"step": [...], "fatigue": [...], "level": [...]}  (logged monitor values)
  memory                    {"items_added", "items_evicted", "replays_wasted", "replays_run"} summed
                            over sleep lines (0 for arms without a memory store)

Rates (identical definitions for every arm, from the raw log, no policy hooks):
  perseveration_rate  fraction of tagged agent steps whose APPROACH_TAGS overlap an EARLIER failed
                      step's tags by >= fatigue.perseveration_match (the FatigueMonitor's rule;
                      failure = fatigue.perseveration_failure)
  invalid_rate        fraction of agent steps with valid = false
  calibration_error   mean |expected_score - score| / |best_known| over valid steps with a prediction
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from evaluators.base import normalized_score
from harness.fatigue import FatigueConfig, SleepDepth, tag_overlap

METRICS_SCHEMA = "baseline_metrics/v1"
LLM_DEPTHS = ("nrem", "nrem_rem")
RATE_KEYS = ("perseveration_rate", "invalid_rate", "calibration_error")


def _finite(x: Any) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return float(x) if math.isfinite(x) else None


def _key(f: float) -> str:
    return f"{float(f):g}"


def step_records(history: Iterable[dict]) -> list[dict]:
    return [h for h in history if h.get("event", "step") == "step"]


def sleep_records(history: Iterable[dict]) -> list[dict]:
    return [h for h in history if h.get("event") == "sleep"]


def sleep_depth_of(rec: dict) -> str:
    """Depth label of a sleep line: sleep_depth, else sleep_mode, else 'nrem' (older logs)."""
    if rec.get("sleep_depth"):
        return SleepDepth.parse(rec["sleep_depth"]).label
    mode = rec.get("sleep_mode")
    if mode in ("rem", "nrem+rem"):
        return "nrem_rem"
    if mode is None and (rec.get("details") or {}).get("llm_calls") == 0:
        return "forget"
    return "nrem"


def llm_sleep_steps(history: Iterable[dict]) -> list[int]:
    """Steps after which an LLM sleep (nrem / nrem_rem) ran: the count matched arms reproduce."""
    return [int(h["step"]) for h in sleep_records(history) if sleep_depth_of(h) in LLM_DEPTHS]


def used_tokens(rec: dict) -> int:
    return int(rec.get("agent_tokens") or 0) + int(rec.get("sleep_tokens") or 0)


def best_curve(history: Iterable[dict], best_known: float, total_budget: int) -> list[tuple[float, int, float | None]]:
    """[(budget_fraction_used, step, best_so_far_normalized)] per step line, in order."""
    out = []
    best: float | None = None
    for h in step_records(history):
        s = _finite(h.get("score")) if h.get("valid") else None
        if s is not None and (best is None or s > best):
            best = s
        out.append((used_tokens(h) / total_budget, int(h.get("step") or 0),
                    normalized_score(best, best_known) if best is not None else None))
    return out


def normalized_at(curve: list[tuple[float, int, float | None]], fraction: float) -> float | None:
    """Best-so-far normalized score once `fraction` of the budget was spent (carried forward)."""
    val = None
    for frac, _, norm in curve:
        if frac <= fraction + 1e-12:
            val = norm
        else:
            break
    return val


def first_reach(curve: list[tuple[float, int, float | None]], target: float) -> dict:
    for frac, step, norm in curve:
        if norm is not None and norm >= target:
            return {"reached": True, "budget_fraction": round(frac, 6), "step": step}
    return {"reached": False, "budget_fraction": None, "step": None}


def perseveration_flags(steps: list[dict], cfg: FatigueConfig | None = None) -> list[bool | None]:
    """Per step record: tags overlap an earlier failed step (None: untagged). The monitor's rule."""
    cfg = cfg or FatigueConfig()
    best: float | None = None
    failed: list[list[str]] = []
    out: list[bool | None] = []
    for h in steps:
        tags = [str(t) for t in (h.get("approach_tags") or [])]
        valid = bool(h.get("valid"))
        s = _finite(h.get("score")) if valid else None
        improved = s is not None and (best is None or s > best + cfg.improvement_min_delta)
        pool = failed if cfg.perseveration_lookback is None else failed[-cfg.perseveration_lookback:]
        out.append(any(tag_overlap(tags, f) >= cfg.perseveration_match for f in pool) if tags else None)
        is_failed = (not valid) if cfg.perseveration_failure == "invalid" else (not valid or not improved)
        if is_failed and tags:
            failed.append(tags)
        if improved:
            best = s
    return out


def rates(steps: list[dict], persev: list[bool | None], best_known: float | None) -> dict:
    """RATES over `steps` (aligned `persev` flags). None where nothing is measurable."""
    scale = abs(best_known) if best_known else 1.0
    pv = [p for p in persev if p is not None]
    gaps = []
    for h in steps:
        e, s = _finite(h.get("expected_score")), _finite(h.get("score"))
        if h.get("valid") and e is not None and s is not None:
            gaps.append(abs(s - e) / scale)
    return {
        "perseveration_rate": (sum(pv) / len(pv)) if pv else None,
        "invalid_rate": (sum(1 for h in steps if not h.get("valid")) / len(steps)) if steps else None,
        "calibration_error": (sum(gaps) / len(gaps)) if gaps else None,
        "n": len(steps),
    }


def _mean_rates(items: list[dict]) -> dict:
    out: dict[str, Any] = {}
    for k in RATE_KEYS:
        vals = [r[k] for r in items if r.get(k) is not None]
        out[k] = (sum(vals) / len(vals)) if vals else None
    out["n"] = len(items)
    return out


def around_steps(steps: list[dict], sleep_steps: Iterable[int], window: int, best_known: float | None,
                 cfg: FatigueConfig | None = None, depths: Iterable[str] | None = None) -> dict:
    """RATES in the `window` agent steps up to (inclusive) and after each reference step.

    Works for any reference steps: a run's own sleeps, or (analysis) another arm's sleep steps
    used as placebo points for an arm that never sleeps."""
    agent = [h for h in steps if (h.get("step") or 0) > 0]
    flags = perseveration_flags(steps, cfg)[len(steps) - len(agent):]
    idx = {int(h.get("step") or 0): i for i, h in enumerate(agent)}
    per = []
    depths = list(depths) if depths is not None else None
    for j, st in enumerate(sleep_steps):
        i = idx.get(int(st))
        if i is None:
            continue
        b0, a1 = max(0, i - window + 1), min(len(agent), i + 1 + window)
        per.append({"step": int(st), "depth": depths[j] if depths else None,
                    "before": rates(agent[b0:i + 1], flags[b0:i + 1], best_known),
                    "after": rates(agent[i + 1:a1], flags[i + 1:a1], best_known)})
    return {"before_sleep": _mean_rates([p["before"] for p in per if p["before"]["n"]]),
            "after_sleep": _mean_rates([p["after"] for p in per if p["after"]["n"]]),
            "around_sleeps": per}


def compute_run_metrics(history: list[dict], *, best_known: float, total_budget: int,
                        budget: dict | None = None, scheduled: int | None = None,
                        cfg: Any = None, fatigue_cfg: FatigueConfig | dict | None = None,
                        arm: str | None = None, sleep_policy: str | None = None) -> dict:
    """The `metrics` block of summary.json (schema in the module docstring).

    `budget` is Budget.snapshot() when available (online); offline it is rebuilt from the last line.
    `cfg` is a harness.config.MetricsConfig (or None for defaults)."""
    from harness.config import MetricsConfig

    mcfg = cfg if cfg is not None else MetricsConfig()
    fcfg = fatigue_cfg if isinstance(fatigue_cfg, FatigueConfig) else FatigueConfig(**(fatigue_cfg or {}))
    steps = step_records(history)
    sleeps = sleep_records(history)
    curve = best_curve(history, best_known, total_budget)
    last = history[-1] if history else {}
    if budget is None:
        budget = {"total_tokens": total_budget, "used": used_tokens(last), "agent_tokens": last.get("agent_tokens") or 0,
                  "sleep_tokens": last.get("sleep_tokens") or 0, "agent_calls": None, "sleep_calls": None,
                  "overshoot": max(0, used_tokens(last) - total_budget)}

    depths = [sleep_depth_of(h) for h in sleeps]
    by_depth = {d: depths.count(d) for d in ("forget", "nrem", "nrem_rem")}
    llm_steps = [int(h["step"]) for h, d in zip(sleeps, depths) if d in LLM_DEPTHS]
    llm_depths = [d for d in depths if d in LLM_DEPTHS]
    agent = [h for h in steps if (h.get("step") or 0) > 0]
    flags = perseveration_flags(steps, fcfg)[len(steps) - len(agent):]
    around = around_steps(steps, llm_steps, mcfg.window, best_known, fcfg, llm_depths)

    used = int(budget.get("used") or 0)
    sleep_tok = int(budget.get("sleep_tokens") or 0)
    final = curve[-1][2] if curve else None
    memory = {k: sum(int(h.get(k) or 0) for h in sleeps) for k in ("items_added", "items_evicted", "replays_wasted")}
    memory["replays_run"] = sum(int((h.get("details") or {}).get("replays_run") or 0) for h in sleeps)
    for k in ("dream_calls", "dream_evaluations", "dreams_validated", "dreams_refuted",
              "dreams_uncertain", "dreams_untested"):
        memory[k] = sum(int((h.get("details") or {}).get(k) or 0) for h in sleeps)

    return {
        "schema": METRICS_SCHEMA,
        "arm": arm,
        "sleep_policy": sleep_policy,
        "best_known": best_known,
        "checkpoints": {_key(f): normalized_at(curve, f) for f in mcfg.checkpoints},
        "reach": {_key(t): first_reach(curve, t) for t in mcfg.thresholds},
        "final_best_normalized": final,
        "budget_used_fraction": round(used / total_budget, 6) if total_budget else None,
        "steps_completed": max((int(h.get("step") or 0) for h in steps), default=0),
        "sleep": {
            "llm_sleeps": len(llm_steps), "forgets": by_depth["forget"], "by_depth": by_depth,
            "llm_sleep_steps": llm_steps,
            "forget_steps": [int(h["step"]) for h, d in zip(sleeps, depths) if d == "forget"],
            "scheduled": scheduled, "realized": len(llm_steps),
            "deferrals": sum(1 for h in steps if h.get("sleep_deferred")),
        },
        "tokens": {
            "total_budget": total_budget, "used": used, "agent": int(budget.get("agent_tokens") or 0),
            "sleep": sleep_tok,
            "sleep_share_of_budget": round(sleep_tok / total_budget, 6) if total_budget else None,
            "sleep_share_of_used": round(sleep_tok / used, 6) if used else None,
            "agent_calls": budget.get("agent_calls"), "sleep_calls": budget.get("sleep_calls"),
            "overshoot": budget.get("overshoot"),
        },
        "behavior": {"window": mcfg.window, "overall": rates(agent, flags, best_known), **around},
        "fatigue_curve": {"step": [h.get("step") for h in steps], "fatigue": [h.get("fatigue") for h in steps],
                          "level": [h.get("fatigue_level") for h in steps]},
        "memory": memory,
    }


__all__ = [
    "LLM_DEPTHS", "METRICS_SCHEMA", "around_steps", "best_curve", "compute_run_metrics", "first_reach",
    "llm_sleep_steps", "normalized_at", "perseveration_flags", "rates", "sleep_depth_of", "step_records",
]
