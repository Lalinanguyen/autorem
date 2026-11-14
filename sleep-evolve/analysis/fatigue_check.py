"""Predictive check: does fatigue at step t predict a bad outcome within the next k steps?

Bad outcome at t (label 1): none of steps t+1..t+k improves the best valid score so far, OR any
of them is invalid. Steps with fewer than k future steps are dropped (no censored labels).
AUC (numpy-only, Mann-Whitney with average ranks for ties) is reported per problem on the
`validation.arm` runs (no_sleep by default), pooling seeds. When |AUC - 0.5| < auc_margin the
result is flagged: the impairment weights need recalibrating from pilot data
(scripts/refit_fatigue_weights.py).

CLI:  uv run python -m analysis.fatigue_check --runs runs [--config configs/example.yaml] [--k 5]
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field

import numpy as np

from analysis.fatigue_logs import RunLog, find_runs, load_fatigue_config, recompute_fatigue
from harness.fatigue import COMPONENTS, FatigueConfig

RECALIBRATE_MSG = ("AUC is within {margin} of 0.5: fatigue does not predict outcomes on this problem; "
                   "recalibrate the impairment weights from pilot data (scripts/refit_fatigue_weights.py)")


def auc(scores, labels) -> float | None:
    """ROC AUC = P(score_pos > score_neg) + 0.5 P(tie). None if a class is empty."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def outcome_labels(steps: list[dict], k: int, *, skip_seed: bool = True,
                   min_delta: float = 0.0) -> list[int | None]:
    """Per step record: 1 = no improvement or an invalid step within the next k steps, 0 = otherwise,
    None = fewer than k future steps (or the seed step when skip_seed)."""
    best = -math.inf
    improved, invalid = [], []
    for s in steps:
        sc = s.get("score")
        ok = bool(s.get("valid")) and isinstance(sc, (int, float)) and math.isfinite(sc)
        imp = ok and sc > best + min_delta
        if imp:
            best = sc
        improved.append(imp)
        invalid.append(not s.get("valid"))
    n = len(steps)
    out: list[int | None] = []
    for i, s in enumerate(steps):
        if (skip_seed and s.get("step") == 0) or i + k >= n:
            out.append(None)
            continue
        fut = range(i + 1, i + k + 1)
        bad = not any(improved[j] for j in fut) or any(invalid[j] for j in fut)
        out.append(int(bad))
    return out


@dataclass
class RunFeatures:
    """Weight-independent per-step features (the components do not depend on the weights)."""

    run_id: str
    problem_id: int | None
    load: np.ndarray        # (n,)
    debt: np.ndarray        # (n,)
    comps: np.ndarray       # (n, len(COMPONENTS)), nan = unavailable
    labels: np.ndarray      # (n,), -1 = unlabeled
    fatigue: np.ndarray     # (n,) under the config the features were computed with


def run_features(run: RunLog, cfg: FatigueConfig, k: int | None = None, *, use_logged: bool = False) -> RunFeatures:
    k = k or cfg.validation.horizon_k
    steps = run.steps
    labels = outcome_labels(steps, k, skip_seed=cfg.skip_seed_step, min_delta=cfg.improvement_min_delta)

    def nan(v):
        return np.nan if v is None else v

    if use_logged and steps and all(s.get("fatigue") is not None for s in steps):
        load = [s.get("fatigue_load") or 0.0 for s in steps]
        debt = [s.get("fatigue_debt") or 0.0 for s in steps]
        comps = [[nan(s.get(f"fatigue_{c}")) for c in COMPONENTS] for s in steps]
        fat = [s["fatigue"] for s in steps]
    else:
        states = recompute_fatigue(run, cfg)
        load = [st.load for st in states]
        debt = [st.debt for st in states]
        comps = [[nan(st.components.get(c)) for c in COMPONENTS] for st in states]
        fat = [st.fatigue for st in states]
    return RunFeatures(run.run_id, run.problem_id, np.asarray(load, float), np.asarray(debt, float),
                       np.asarray(comps, float).reshape(len(steps), len(COMPONENTS)),
                       np.asarray([-1 if v is None else v for v in labels], int), np.asarray(fat, float))


def fatigue_from_features(f: RunFeatures, weights, cfg: FatigueConfig) -> np.ndarray:
    """Fatigue recomputed from features under impairment `weights` (order: COMPONENTS).

    Matches FatigueMonitor exactly (same weighted mean, renormalization and mix)."""
    w = np.asarray(weights, float)
    avail = ~np.isnan(f.comps)
    vals = np.where(avail, f.comps, 0.0)
    pos = w > 0
    if cfg.renormalize_missing:
        den = (avail * pos * w).sum(axis=1)
    else:
        den = np.full(len(vals), float((w * pos).sum()))
    num = (vals * w * pos).sum(axis=1)
    imp = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return cfg.load_weight * f.load + cfg.impairment_weight * imp + f.debt


@dataclass
class ProblemAUC:
    problem_id: int | None
    auc: float | None
    n: int
    n_pos: int
    runs: list[str] = field(default_factory=list)
    flagged: bool = False
    message: str | None = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def problem_aucs(features: list[RunFeatures], cfg: FatigueConfig, weights=None) -> list[ProblemAUC]:
    """AUC per problem (seeds pooled). `weights` overrides the config's impairment weights."""
    v = cfg.validation
    by_problem: dict = {}
    for f in features:
        by_problem.setdefault(f.problem_id, []).append(f)
    out = []
    for pid in sorted(by_problem, key=lambda p: (p is None, p if p is not None else 0)):
        fs = by_problem[pid]
        xs, ys = [], []
        for f in fs:
            fat = f.fatigue if weights is None else fatigue_from_features(f, weights, cfg)
            m = f.labels >= 0
            xs.append(fat[m])
            ys.append(f.labels[m])
        x = np.concatenate(xs) if xs else np.zeros(0)
        y = np.concatenate(ys) if ys else np.zeros(0, int)
        a = auc(x, y) if len(y) >= v.min_samples else None
        r = ProblemAUC(pid, a, int(len(y)), int(y.sum()), [f.run_id for f in fs])
        if a is None:
            r.flagged = True
            r.message = f"AUC not computable (n={len(y)}, positives={int(y.sum())}); collect more pilot data"
        elif abs(a - 0.5) < v.auc_margin:
            r.flagged = True
            r.message = RECALIBRATE_MSG.format(margin=v.auc_margin)
        out.append(r)
    return out


def predictive_check(runs: list[RunLog], cfg: FatigueConfig | None = None, *, k: int | None = None,
                     arm: str | None = None, use_logged: bool = False) -> list[ProblemAUC]:
    """AUC per problem on the `arm` runs (default cfg.validation.arm; pass arm='' for all runs)."""
    cfg = cfg or FatigueConfig()
    arm = cfg.validation.arm if arm is None else arm
    sel = [r for r in runs if not arm or r.policy == arm]
    feats = [run_features(r, cfg, k, use_logged=use_logged) for r in sel]
    return problem_aucs(feats, cfg)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fatigue predictive check (AUC per problem)")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--config", default=None, help="run config YAML; its fatigue: section is used")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--use-logged", action="store_true", help="use logged fatigue instead of recomputing")
    a = ap.parse_args(argv)
    cfg = load_fatigue_config(a.config)
    res = predictive_check(find_runs(a.runs), cfg, k=a.k, arm=a.arm, use_logged=a.use_logged)
    print(json.dumps([r.to_dict() for r in res], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
