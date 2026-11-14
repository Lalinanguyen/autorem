"""Baseline comparison: does sleep help (Q1), and does fatigue-triggered sleep beat fixed schedules (Q2)?

Reads every run under --runs (history.jsonl + summary.json, written by harness/loop.py for the arms of
harness/baselines.py). Every comparison is PAIRED by (problem, seed). Per-run metrics come from
summary.json["metrics"] (harness/metrics.py) and are recomputed from history.jsonl when absent.

Contrasts (first arm minus second arm, per pair):
  Q1  fatigue - no_sleep, fixed_letta - no_sleep (and sleep_lite - no_sleep when present)
  Q2  fatigue - fixed_letta, fatigue - fixed_matched (the key timing test),
      fixed_matched - random_matched (does fixed timing beat random?)
Outcomes:
  budget_to_95   fraction of the budget used to first reach 95% of best_known (lower is better).
                 "Not reached" is censored; it enters the paired difference as 1.0 (the whole budget,
                 a lower bound on the true value), which can only understate a difference, and the
                 number of censored runs is reported.
  final_score    final best normalized score (higher is better)
  score_at_25    best normalized score after 25% of the budget (higher is better)
Per contrast and outcome: median paired difference with a seeded bootstrap 95% CI, two-sided
Wilcoxon signed-rank p (analysis/baseline_stats.py: exact for <= 20 non-zero pairs, else normal
approximation), and win/tie/loss counts from the first arm's point of view.

Cost: per arm, the total and mean dollar cost of its runs from summary.json["cost"]["total"]
(harness/cost.py, computed from the logged usage and the config's price table); runs without a known
cost (mock configs, older logs, unpriced models) are counted in `cost_unknown_runs` and left out of
the sums, so a total over a partly unknown arm is marked with "+" in the report.

Figures (matplotlib, optional `analysis` group, imported lazily):
  a  score_vs_budget.png        best normalized score vs budget used: median + IQR band per arm, per problem
  b  fatigue_curves.png         fatigue over steps, no_sleep vs fatigue, sleep events marked
  c  perseveration_sleep.png    perseveration (also invalid rate, calibration error) before vs after sleep,
                                per arm; no_sleep is measured at the paired fatigue run's sleep steps (placebo)
  d  sleep_count_vs_gain.png    LLM sleep count vs gain over the paired no_sleep run

CLI:
  uv run --group analysis python -m analysis.compare_baselines --runs runs/baselines_mock --out runs/baselines_mock/report
  (--no-plot to skip figures; writes compare_baselines.json and report.txt in --out and prints the report)
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from analysis.baseline_stats import bootstrap_median_ci, wilcoxon_signed_rank, win_tie_loss
from analysis.fatigue_logs import RunLog, find_runs
from harness.fatigue import FatigueConfig
from harness.metrics import around_steps, best_curve, compute_run_metrics, normalized_at

ALPHA = 0.05
ARM_ORDER = ("no_sleep", "sleep_lite", "fixed_letta", "fixed_sparse", "fixed_matched", "random_matched",
             "fatigue", "fixed_tiered")
SLEEP_ARMS = ("fixed_letta", "fixed_sparse", "fixed_matched", "random_matched", "fatigue", "fixed_tiered")
CONTRASTS = (
    ("Q1", "fatigue", "no_sleep"),
    ("Q1", "fixed_letta", "no_sleep"),
    ("Q1", "sleep_lite", "no_sleep"),
    ("Q2", "fatigue", "fixed_letta"),
    ("Q2", "fatigue", "fixed_matched"),
    ("Q2", "fixed_matched", "random_matched"),
)
OUTCOMES = {  # name -> (label, better)
    "budget_to_95": ("budget fraction to reach 95% of best_known", "lower"),
    "final_score": ("final best normalized score", "higher"),
    "score_at_25": ("best normalized score at 25% of budget", "higher"),
}
PRIMARY = "final_score"
BUDGET_GRID = np.linspace(0.0, 1.0, 51)

INTERPRETATION_GUIDE = """Interpretation guide
  - fatigue > fixed_matched                      -> timing matters, beyond the amount of sleep
  - fatigue ~ fixed_matched, both > fixed_letta  -> the gain comes from sleeping less often, not from smart timing
  - fixed_matched ~ random_matched               -> fixed schedules carry no timing benefit
  - any sleep arm < no_sleep                     -> consolidation costs more budget than it returns at this run length
  ('>' / '<' = significant paired difference at alpha; '~' = not significant)"""


@dataclass
class BaselineRun:
    run: RunLog
    arm: str
    metrics: dict
    curve: list = field(default_factory=list)

    @property
    def key(self) -> tuple:
        return (self.run.problem_id, self.run.seed)

    def outcome(self, name: str) -> tuple[float | None, bool]:
        """(value, censored)."""
        m = self.metrics
        if name == "final_score":
            return m.get("final_best_normalized"), False
        if name == "score_at_25":
            return (m.get("checkpoints") or {}).get("0.25"), False
        if name == "budget_to_95":
            r = (m.get("reach") or {}).get("0.95") or {}
            if r.get("reached"):
                return r.get("budget_fraction"), False
            return 1.0, True
        raise KeyError(name)


def arm_of(run: RunLog) -> str:
    s = run.summary or {}
    return str(s.get("arm") or (s.get("config") or {}).get("baseline_arm") or run.policy)


def _fatigue_cfg(run: RunLog) -> FatigueConfig:
    try:
        return FatigueConfig(**((run.summary.get("config") or {}).get("fatigue") or {}))
    except Exception:
        return FatigueConfig()


def load_baseline_runs(root: str | Path) -> list[BaselineRun]:
    out = []
    for r in find_runs(root):
        total = r.total_budget
        if not total or r.best_known is None:
            continue
        m = (r.summary or {}).get("metrics")
        if not m or "error" in m:
            m = compute_run_metrics(r.history, best_known=r.best_known, total_budget=total,
                                    budget=(r.summary or {}).get("budget"), fatigue_cfg=_fatigue_cfg(r),
                                    arm=arm_of(r), sleep_policy=r.policy)
        out.append(BaselineRun(r, arm_of(r), m, best_curve(r.history, r.best_known, total)))
    return out


def pair_table(runs: list[BaselineRun]) -> dict[tuple, dict[str, BaselineRun]]:
    pairs: dict[tuple, dict[str, BaselineRun]] = {}
    for br in runs:
        pairs.setdefault(br.key, {})[br.arm] = br
    return pairs


def _arms_present(runs: list[BaselineRun]) -> list[str]:
    present = {r.arm for r in runs}
    return [a for a in ARM_ORDER if a in present] + sorted(present - set(ARM_ORDER))


# ---------------------------------------------------------------------------------------------
# Contrasts
# ---------------------------------------------------------------------------------------------


def contrast(pairs: dict, a: str, b: str, outcome: str, *, n_boot: int = 10000, seed: int = 0,
             alpha: float = ALPHA) -> dict:
    label, better = OUTCOMES[outcome]
    diffs, keys, censored = [], [], 0
    for key in sorted(pairs, key=lambda k: (str(k[0]), str(k[1]))):
        arms = pairs[key]
        if a not in arms or b not in arms:
            continue
        va, ca = arms[a].outcome(outcome)
        vb, cb = arms[b].outcome(outcome)
        if va is None or vb is None:
            continue
        censored += int(ca) + int(cb)
        diffs.append(float(va) - float(vb))
        keys.append(list(key))
    med, lo, hi = bootstrap_median_ci(diffs, n_boot=n_boot, seed=seed)
    w = wilcoxon_signed_rank(diffs)
    sign = 1.0 if better == "higher" else -1.0
    sig = len(diffs) > 0 and w["p"] < alpha
    if not diffs:
        verdict = "no pairs"
    elif not sig:
        verdict = "not significant"
    else:
        verdict = f"{a} better" if (med or 0.0) * sign > 0 else f"{b} better"
    return {"a": a, "b": b, "outcome": outcome, "label": label, "better": better, "n_pairs": len(diffs),
            "pairs": keys, "diffs": diffs, "median_diff": med, "ci95": [lo, hi], "p": w["p"],
            "test": w["method"], "n_zero": w["n_zero"], **win_tie_loss(diffs, better),
            "censored_runs": censored, "significant": bool(sig), "verdict": verdict}


def all_contrasts(runs: list[BaselineRun], **kw) -> list[dict]:
    pairs = pair_table(runs)
    present = {r.arm for r in runs}
    out = []
    for q, a, b in CONTRASTS:
        if a not in present or b not in present:
            continue
        for outcome in OUTCOMES:
            out.append({"question": q, **contrast(pairs, a, b, outcome, **kw)})
    return out


# ---------------------------------------------------------------------------------------------
# Per-arm summaries and behavior around sleeps
# ---------------------------------------------------------------------------------------------


def _nanmedian(xs) -> float | None:
    v = [x for x in xs if x is not None and math.isfinite(x)]
    return float(np.median(v)) if v else None


def _nanmean(xs) -> float | None:
    v = [x for x in xs if x is not None and math.isfinite(x)]
    return float(np.mean(v)) if v else None


def run_cost(br: BaselineRun) -> float | None:
    """Dollar cost of a run (summary.json["cost"]["total"]); None if unknown."""
    c = ((br.run.summary or {}).get("cost") or {}).get("total")
    return float(c) if isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c) else None


def run_evals(br: BaselineRun) -> tuple[float | None, float | None, float | None]:
    """(used, agent_evals, sleep_evals) from summary.json["evaluations"]; None when unrecorded."""
    e = (br.run.summary or {}).get("evaluations") or {}
    def num(k):
        v = e.get(k)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
    return num("used"), num("agent_evals"), num("sleep_evals")


def arm_summaries(runs: list[BaselineRun]) -> dict[str, dict]:
    out = {}
    for arm in _arms_present(runs):
        rs = [r for r in runs if r.arm == arm]
        sl = [r.metrics.get("sleep") or {} for r in rs]
        tok = [r.metrics.get("tokens") or {} for r in rs]
        mem = [r.metrics.get("memory") or {} for r in rs]
        reach = [(r.metrics.get("reach") or {}).get("0.95") or {} for r in rs]
        out[arm] = {
            "n_runs": len(rs),
            "final_score_median": _nanmedian([r.outcome("final_score")[0] for r in rs]),
            "score_at_25_median": _nanmedian([r.outcome("score_at_25")[0] for r in rs]),
            "reached_95": sum(1 for x in reach if x.get("reached")),
            "budget_to_95_median_reached": _nanmedian([x.get("budget_fraction") for x in reach if x.get("reached")]),
            "llm_sleeps_mean": _nanmean([s.get("llm_sleeps") for s in sl]),
            "forgets_mean": _nanmean([s.get("forgets") for s in sl]),
            "scheduled_vs_realized": [[s.get("scheduled"), s.get("realized")] for s in sl
                                      if s.get("scheduled") is not None],
            "sleep_share_of_budget_mean": _nanmean([t.get("sleep_share_of_budget") for t in tok]),
            "items_added_mean": _nanmean([m.get("items_added") for m in mem]),
            "items_evicted_mean": _nanmean([m.get("items_evicted") for m in mem]),
            "replays_wasted_mean": _nanmean([m.get("replays_wasted") for m in mem]),
            "evals_used_mean": _nanmean([run_evals(r)[0] for r in rs]),
            "evals_agent_mean": _nanmean([run_evals(r)[1] for r in rs]),
            "evals_sleep_mean": _nanmean([run_evals(r)[2] for r in rs]),
            **_cost_fields([run_cost(r) for r in rs]),
        }
    return out


def _cost_fields(costs: list[float | None]) -> dict:
    known = [c for c in costs if c is not None]
    return {"cost_total": round(sum(known), 6) if known else None,
            "cost_mean": round(sum(known) / len(known), 6) if known else None,
            "cost_unknown_runs": len(costs) - len(known)}


def behavior_around_sleep(runs: list[BaselineRun]) -> dict[str, dict]:
    """Mean RATES before / after sleep per arm. no_sleep (and sleep_lite, which never makes an LLM
    sleep) are measured at the paired fatigue run's LLM-sleep steps as placebo reference points."""
    pairs = pair_table(runs)
    out: dict[str, dict] = {}
    keys = ("perseveration_rate", "invalid_rate", "calibration_error")
    for arm in _arms_present(runs):
        before: dict[str, list] = {k: [] for k in keys}
        after: dict[str, list] = {k: [] for k in keys}
        reference = "own LLM sleeps"
        for r in (x for x in runs if x.arm == arm):
            b = r.metrics.get("behavior") or {}
            if arm in ("no_sleep", "sleep_lite"):
                src = pairs.get(r.key, {}).get("fatigue")
                if src is None:
                    continue
                ref_steps = (src.metrics.get("sleep") or {}).get("llm_sleep_steps") or []
                b = around_steps(r.run.steps, ref_steps, (r.metrics.get("behavior") or {}).get("window", 5),
                                 r.run.best_known, _fatigue_cfg(r.run))
                reference = "paired fatigue run's sleep steps (placebo)"
            for k in keys:
                before[k].append((b.get("before_sleep") or {}).get(k))
                after[k].append((b.get("after_sleep") or {}).get(k))
        out[arm] = {"reference": reference,
                    "before": {k: _nanmean(v) for k, v in before.items()},
                    "after": {k: _nanmean(v) for k, v in after.items()}}
    return out


# ---------------------------------------------------------------------------------------------
# Plain-language summary and interpretation
# ---------------------------------------------------------------------------------------------


def _fmt(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:+.{nd}f}"


def _find(cs: list[dict], a: str, b: str, outcome: str = PRIMARY) -> dict | None:
    return next((c for c in cs if c["a"] == a and c["b"] == b and c["outcome"] == outcome), None)


def _sentence(c: dict | None) -> str:
    if c is None or not c["n_pairs"]:
        return None
    sig = "significant" if c["significant"] else "NOT significant"
    return (f"{c['a']} vs {c['b']}: median paired difference in {c['label']} {_fmt(c['median_diff'])} "
            f"(95% CI {_fmt(c['ci95'][0])} to {_fmt(c['ci95'][1])}), Wilcoxon p={c['p']:.3g} ({sig}), "
            f"W/T/L {c['win']}/{c['tie']}/{c['loss']} over {c['n_pairs']} pair{'s' if c['n_pairs'] != 1 else ''}")


def _better(c: dict | None) -> int:
    """+1: first arm significantly better, -1: significantly worse, 0: not significant / missing."""
    if c is None or not c["significant"] or c["median_diff"] is None:
        return 0
    sign = 1 if c["better"] == "higher" else -1
    return 1 if c["median_diff"] * sign > 0 else (-1 if c["median_diff"] * sign < 0 else 0)


MIN_PAIRS_FOR_SIMILARITY = 6  # below this no exact two-sided Wilcoxon p can be < 0.05


def interpret(cs: list[dict]) -> list[str]:
    """Which lines of the interpretation guide the primary outcome supports.

    '~' (not significant) is only read as "about equal" with >= MIN_PAIRS_FOR_SIMILARITY pairs;
    with fewer, a non-significant result says nothing and is reported as underpowered."""
    out = []
    n = max((c["n_pairs"] for c in cs), default=0)
    if n < MIN_PAIRS_FOR_SIMILARITY:
        return [f"Underpowered: {n} pair(s) per contrast; no '~' or '>' pattern of the guide can be assessed "
                f"(need >= {MIN_PAIRS_FOR_SIMILARITY} pairs for p < 0.05 to be attainable)."]
    fm, fl = _find(cs, "fatigue", "fixed_matched"), _find(cs, "fatigue", "fixed_letta")
    mr = _find(cs, "fixed_matched", "random_matched")
    if fm is not None and _better(fm) > 0:
        out.append("fatigue > fixed_matched: timing matters, beyond the amount of sleep.")
    if fm is not None and fm["n_pairs"] and _better(fm) == 0 and _better(fl) > 0:
        out.append("fatigue ~ fixed_matched and fatigue > fixed_letta: the gain comes from sleeping less often, "
                   "not from smart timing (check fixed_matched > fixed_letta too).")
    if mr is not None and mr["n_pairs"] and _better(mr) == 0:
        out.append("fixed_matched ~ random_matched: fixed schedules carry no timing benefit.")
    for c in cs:
        if c["outcome"] == PRIMARY and c["b"] == "no_sleep" and _better(c) < 0:
            out.append(f"{c['a']} < no_sleep: consolidation costs more budget than it returns at this run length.")
    if not out:
        out.append("No guide pattern is supported at this alpha: no conclusive difference on the primary outcome.")
    return out


def summary_paragraph(cs: list[dict], alpha: float = ALPHA) -> str:
    q1 = [s for s in (_sentence(_find(cs, a, "no_sleep")) for a in ("fatigue", "fixed_letta", "sleep_lite")) if s]
    q2 = [s for s in (_sentence(_find(cs, a, b)) for a, b in
                      (("fatigue", "fixed_letta"), ("fatigue", "fixed_matched"), ("fixed_matched", "random_matched")))
          if s]
    parts = []
    if q1:
        wins = [c for c in (_find(cs, a, "no_sleep") for a in ("fatigue", "fixed_letta")) if _better(c) > 0]
        losses = [c for c in (_find(cs, a, "no_sleep") for a in ("fatigue", "fixed_letta")) if _better(c) < 0]
        head = ("Q1 (does sleep help?): yes for " + ", ".join(c["a"] for c in wins) if wins else
                "Q1 (does sleep help?): sleep is worse than no sleep for " + ", ".join(c["a"] for c in losses)
                if losses else "Q1 (does sleep help?): no significant difference from no_sleep")
        parts.append(head + ". " + "; ".join(q1) + ".")
    else:
        parts.append("Q1 (does sleep help?): not testable (fatigue / fixed_letta / no_sleep runs missing).")
    if q2:
        fm = _find(cs, "fatigue", "fixed_matched")
        b = _better(fm)
        head = ("Q2 (does fatigue timing beat fixed schedules?): " +
                ("yes, fatigue beats the count-matched fixed schedule" if b > 0 else
                 "no, the count-matched fixed schedule does better" if b < 0 else
                 "not shown: fatigue vs fixed_matched is not significant" if fm and fm["n_pairs"] else
                 "fatigue vs fixed_matched not available"))
        parts.append(head + ". " + "; ".join(q2) + ".")
    else:
        parts.append("Q2 (does fatigue timing beat fixed schedules?): not testable (arms missing).")
    n = max((c["n_pairs"] for c in cs), default=0)
    note = (f" Primary outcome: {OUTCOMES[PRIMARY][0]}; significance at alpha={alpha}, "
            f"two-sided.")
    if 0 < n < 6:
        note += (f" With only {n} pair{'s' if n != 1 else ''} no exact Wilcoxon test can reach p<0.05 (minimum p = 2/2^n), so every "
                 f"result here is not significant by construction; run more seeds.")
    return " ".join(parts) + note


# ---------------------------------------------------------------------------------------------
# Figures (matplotlib imported lazily)
# ---------------------------------------------------------------------------------------------

PALETTE = {"no_sleep": "#5c6773", "sleep_lite": "#c9a227", "fixed_letta": "#e0693b", "fixed_sparse": "#a35bd6",
           "fixed_matched": "#2f9e6f", "random_matched": "#8fb339", "fatigue": "#2a6fdb", "fixed_tiered": "#d6458f"}


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:  # optional dependency
        raise RuntimeError("plotting needs matplotlib (optional `analysis` dependency group: "
                           "uv run --group analysis ...)") from e
    return plt


def _color(arm: str) -> str:
    return PALETTE.get(arm, "#333333")


def _problems(runs: list[BaselineRun]) -> list:
    return sorted({r.run.problem_id for r in runs}, key=lambda p: (p is None, p or 0))


def plot_score_vs_budget(runs: list[BaselineRun], path: Path) -> Path:
    plt = _plt()
    probs = _problems(runs)
    fig, axes = plt.subplots(1, len(probs), figsize=(6.5 * len(probs), 4.5), squeeze=False)
    for ax, pid in zip(axes[0], probs):
        for arm in _arms_present([r for r in runs if r.run.problem_id == pid]):
            rows = [[np.nan if (v := normalized_at(r.curve, f)) is None else v for f in BUDGET_GRID]
                    for r in runs if r.run.problem_id == pid and r.arm == arm]
            if not rows:
                continue
            a = np.asarray(rows, float)
            with warnings.catch_warnings():  # all-NaN columns (no valid program yet) are fine
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(a, axis=0)
                q1, q3 = np.nanpercentile(a, 25, axis=0), np.nanpercentile(a, 75, axis=0)
            ax.plot(BUDGET_GRID, med, color=_color(arm), lw=1.8, label=f"{arm} (n={len(rows)})")
            ax.fill_between(BUDGET_GRID, q1, q3, color=_color(arm), alpha=0.15, lw=0)
        ax.set_xlabel("fraction of token budget used (agent + sleep)")
        ax.set_ylabel("best normalized score (1.0 = best known)")
        ax.set_title(f"Problem {pid}: median and IQR across seeds")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_fatigue_curves(runs: list[BaselineRun], path: Path) -> Path:
    plt = _plt()
    sel = [r for r in runs if r.arm in ("no_sleep", "fatigue")]
    probs = _problems(sel) or [None]
    fig, axes = plt.subplots(1, len(probs), figsize=(7 * len(probs), 4.5), squeeze=False)
    for ax, pid in zip(axes[0], probs):
        seen = set()
        for r in sorted((x for x in sel if x.run.problem_id == pid), key=lambda x: (x.arm, x.run.seed or 0)):
            steps = r.run.steps
            x = np.asarray([h.get("step") for h in steps], float)
            y = np.asarray([np.nan if h.get("fatigue") is None else h["fatigue"] for h in steps], float)
            ax.plot(x, y, color=_color(r.arm), lw=1.4, alpha=0.8, label=r.arm if r.arm not in seen else None)
            seen.add(r.arm)
            sl = r.metrics.get("sleep") or {}
            for key, marker, lab in (("llm_sleep_steps", "v", "LLM sleep"), ("forget_steps", "x", "forget")):
                st = sl.get(key) or []
                if st and len(x):
                    ax.scatter(st, np.interp(st, x, np.nan_to_num(y)), marker=marker, s=30, color=_color(r.arm),
                               zorder=3, label=lab if (lab not in seen) else None)
                    seen.add(lab)
        ax.set_xlabel("agent step")
        ax.set_ylabel("fatigue (shared monitor)")
        ax.set_title(f"Problem {pid}: fatigue, no_sleep vs fatigue (v = LLM sleep, x = forget)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_perseveration(behavior: dict[str, dict], path: Path) -> Path:
    plt = _plt()
    arms = list(behavior)
    keys = (("perseveration_rate", "perseveration rate"), ("invalid_rate", "invalid rate"),
            ("calibration_error", "calibration error"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    x = np.arange(len(arms))
    for ax, (k, title) in zip(axes, keys):
        b = [np.nan if behavior[a]["before"].get(k) is None else behavior[a]["before"][k] for a in arms]
        af = [np.nan if behavior[a]["after"].get(k) is None else behavior[a]["after"][k] for a in arms]
        ax.bar(x - 0.2, b, 0.4, color="#9aa5b1", label="before sleep")
        ax.bar(x + 0.2, af, 0.4, color=[_color(a) for a in arms], label="after sleep")
        ax.set_xticks(x)
        ax.set_xticklabels([a + ("*" if "placebo" in behavior[a]["reference"] else "") for a in arms],
                           rotation=30, ha="right", fontsize=8)
        ax.set_title(title + (" (main)" if k == "perseveration_rate" else ""))
        top = np.nanmax(np.r_[b, af, 0.0]) if np.isfinite(np.r_[b, af]).any() else 0.0
        ax.set_ylim(0.0, top * 1.1 if top > 0 else 1.0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Behavior in the window before vs after each LLM sleep (* = at the paired fatigue run's sleep steps)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_sleep_vs_gain(runs: list[BaselineRun], path: Path) -> Path:
    plt = _plt()
    pairs = pair_table(runs)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    have_base = any("no_sleep" in v for v in pairs.values())
    rng = np.random.default_rng(0)
    seen = set()
    for r in runs:
        if r.arm == "no_sleep":
            continue
        final = r.outcome("final_score")[0]
        if final is None:
            continue
        base = pairs.get(r.key, {}).get("no_sleep")
        if have_base:
            if base is None or base.outcome("final_score")[0] is None:
                continue
            y = final - base.outcome("final_score")[0]
        else:
            y = final
        n = (r.metrics.get("sleep") or {}).get("llm_sleeps") or 0
        ax.scatter(n + rng.uniform(-0.15, 0.15), y, color=_color(r.arm), s=34, alpha=0.85,
                   label=r.arm if r.arm not in seen else None)
        seen.add(r.arm)
    if have_base:
        ax.axhline(0.0, color="#999999", lw=0.8, ls=":")
    ax.set_xlabel("LLM sleeps in the run (nrem / nrem_rem)")
    ax.set_ylabel("final normalized score - paired no_sleep" if have_base else "final normalized score")
    ax.set_title("Sleep count vs gain (matched arms copy the fatigue count)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


FIGURES = ("score_vs_budget.png", "fatigue_curves.png", "perseveration_sleep.png", "sleep_count_vs_gain.png")


def make_figures(runs: list[BaselineRun], behavior: dict, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return [plot_score_vs_budget(runs, out / FIGURES[0]), plot_fatigue_curves(runs, out / FIGURES[1]),
            plot_perseveration(behavior, out / FIGURES[2]), plot_sleep_vs_gain(runs, out / FIGURES[3])]


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------


def compare(root: str | Path, *, n_boot: int = 10000, seed: int = 0, alpha: float = ALPHA) -> dict:
    runs = load_baseline_runs(root)
    cs = all_contrasts(runs, n_boot=n_boot, seed=seed, alpha=alpha)
    pairs = pair_table(runs)
    return {
        "runs": len(runs), "pairs": len(pairs), "arms": _arms_present(runs), "alpha": alpha,
        "n_boot": n_boot, "bootstrap_seed": seed,
        "arm_summaries": arm_summaries(runs),
        "cost": _cost_fields([run_cost(r) for r in runs]),
        "behavior_around_sleep": behavior_around_sleep(runs),
        "contrasts": cs,
        "summary": summary_paragraph(cs, alpha),
        "interpretation": interpret(cs),
        "interpretation_guide": INTERPRETATION_GUIDE,
        "_runs": runs,
    }


def format_report(rep: dict) -> str:
    lines = [f"Baseline comparison: {rep['runs']} runs, {rep['pairs']} (problem, seed) pairs, arms: "
             f"{', '.join(rep['arms'])}", ""]
    lines.append("Per arm (medians across runs; sleeps = LLM sleeps):")
    lines.append(f"  {'arm':16s} {'n':>3s} {'final':>7s} {'@25%':>7s} {'95%reach':>8s} {'b2_95':>6s} "
                 f"{'sleeps':>6s} {'forget':>6s} {'sleep%':>6s} {'added':>6s} {'evict':>6s} {'wasted':>6s} "
                 f"{'evals':>6s} {'ev/slp':>6s} {'cost$':>9s}")
    f = (lambda x, nd=3: "-" if x is None else f"{x:.{nd}f}")
    for arm, s in rep["arm_summaries"].items():
        lines.append(f"  {arm:16s} {s['n_runs']:>3d} {f(s['final_score_median']):>7s} {f(s['score_at_25_median']):>7s} "
                     f"{s['reached_95']:>4d}/{s['n_runs']:<3d} {f(s['budget_to_95_median_reached'], 2):>6s} "
                     f"{f(s['llm_sleeps_mean'], 1):>6s} {f(s['forgets_mean'], 1):>6s} "
                     f"{f(None if s['sleep_share_of_budget_mean'] is None else 100 * s['sleep_share_of_budget_mean'], 1):>6s} "
                     f"{f(s['items_added_mean'], 1):>6s} {f(s['items_evicted_mean'], 1):>6s} "
                     f"{f(s['replays_wasted_mean'], 1):>6s} {f(s['evals_used_mean'], 1):>6s} "
                     f"{f(s['evals_sleep_mean'], 1):>6s} {_cost_txt(s):>9s}")
        if s["scheduled_vs_realized"]:
            lines.append(f"  {'':16s} scheduled vs realized sleeps per run: {s['scheduled_vs_realized']}")
    if rep.get("cost"):
        lines.append(f"  total cost of all runs: {_cost_txt(rep['cost'])} (USD, from logged usage; "
                     "'+' = some runs have no known cost)")
    lines += ["", "Behavior before -> after LLM sleeps (mean rates; * = placebo at the paired fatigue run's sleep steps):"]
    for arm, b in rep["behavior_around_sleep"].items():
        star = "*" if "placebo" in b["reference"] else " "
        parts = [f"{k.replace('_rate', '').replace('_error', ' err')} {f(b['before'][k])}->{f(b['after'][k])}"
                 for k in ("perseveration_rate", "invalid_rate", "calibration_error")]
        lines.append(f"  {arm + star:17s} " + "  ".join(parts))
    lines += ["", "Paired contrasts (first arm minus second; W/T/L from the first arm's view):"]
    lines.append(f"  {'Q':2s} {'contrast':32s} {'outcome':13s} {'n':>3s} {'median':>8s} {'95% CI':>19s} "
                 f"{'p':>7s} {'W/T/L':>8s} {'cens':>4s}  verdict")
    for c in rep["contrasts"]:
        ci = "-" if c["ci95"][0] is None else f"[{c['ci95'][0]:+.3f},{c['ci95'][1]:+.3f}]"
        med = "-" if c["median_diff"] is None else f"{c['median_diff']:+.4f}"
        lines.append(f"  {c['question']:2s} {c['a'] + ' - ' + c['b']:32s} {c['outcome']:13s} {c['n_pairs']:>3d} "
                     f"{med:>8s} {ci:>19s} {c['p']:>7.3g} {c['win']:>2d}/{c['tie']}/{c['loss']:<3d} "
                     f"{c['censored_runs']:>4d}  {c['verdict']}")
    test = next((c["test"] for c in rep["contrasts"] if c["n_pairs"]), "n/a")
    lines += [f"  (Wilcoxon signed-rank, two-sided: {test}; median CI: percentile bootstrap, "
              f"{rep['n_boot']} resamples, seed {rep['bootstrap_seed']}; budget_to_95 'not reached' = 1.0, censored)",
              "", "Summary", "  " + rep["summary"], "", "What the results support:"]
    lines += [f"  - {x}" for x in rep["interpretation"]]
    lines += ["", rep["interpretation_guide"]]
    return "\n".join(lines)


def _cost_txt(s: dict) -> str:
    tot = s.get("cost_total")
    if tot is None:
        return "-"
    return f"{tot:.4f}" + ("+" if s.get("cost_unknown_runs") else "")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Paired baseline comparison (Q1: sleep vs none, Q2: fatigue vs fixed)")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="baselines_report")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(argv)
    rep = compare(a.runs, n_boot=a.n_boot, seed=a.seed, alpha=a.alpha)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    text = format_report(rep)
    runs = rep.pop("_runs")
    (out / "compare_baselines.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    (out / "report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    if not a.no_plot:
        for p in make_figures(runs, rep["behavior_around_sleep"], out):
            print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
