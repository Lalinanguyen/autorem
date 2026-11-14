"""Refit the fatigue impairment weights from pilot logs to maximize the predictive AUC.

Objective: mean over problems of the per-problem AUC (analysis/fatigue_check.py) of fatigue at
step t for "no improvement or an invalid step within the next k steps", on the validation arm
(no_sleep by default). The impairment components do not depend on the weights, so they are
computed once per run; the search then only re-mixes them (numpy only).

Search (seeded, deterministic): `--samples` random points on the simplex (Dirichlet(1)), plus the
current weights and every vertex, then coordinate ascent on the best point with a shrinking step.
In-sample AUC is optimistic; hold out seeds (--runs on a pilot directory) before trusting it.

Writes the weights into the `fatigue.impairment_weights` section of a config YAML: a new file
(--out, default <config stem>.refit.yaml) or the given config with --in-place (comments are not
preserved by the YAML writer).

  uv run python -m scripts.refit_fatigue_weights --runs runs/pilot --config configs/example.yaml --in-place
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from analysis.fatigue_check import problem_aucs, run_features
from analysis.fatigue_logs import find_runs
from harness.fatigue import COMPONENTS, FatigueConfig


def objective(features, cfg: FatigueConfig, w: np.ndarray) -> float:
    aucs = [r.auc for r in problem_aucs(features, cfg, w) if r.auc is not None]
    return float(np.mean(aucs)) if aucs else float("nan")


def refit(features, cfg: FatigueConfig, *, seed: int = 0, samples: int = 200, rounds: int = 30,
          step0: float = 0.2, min_step: float = 1e-3) -> tuple[np.ndarray, float, float]:
    """Return (weights on the simplex, fitted objective, objective of the config's current weights)."""
    rng = np.random.default_rng(seed)
    d = len(COMPONENTS)
    cur = np.array([cfg.impairment_weights.as_dict()[c] for c in COMPONENTS], float)
    cur = cur / cur.sum() if cur.sum() > 0 else np.full(d, 1.0 / d)
    base = objective(features, cfg, cur)
    cands = [cur, np.full(d, 1.0 / d), *np.eye(d), *rng.dirichlet(np.ones(d), size=samples)]
    best_w, best = cur, base if np.isfinite(base) else -np.inf
    for w in cands:
        v = objective(features, cfg, w)
        if np.isfinite(v) and v > best + 1e-12:
            best_w, best = w, v
    step = step0
    for _ in range(rounds):
        improved = False
        for i in range(d):
            for sgn in (+1.0, -1.0):
                w = best_w.copy()
                w[i] = max(0.0, w[i] + sgn * step)
                if w.sum() <= 0:
                    continue
                w = w / w.sum()
                v = objective(features, cfg, w)
                if np.isfinite(v) and v > best + 1e-12:
                    best_w, best, improved = w, v, True
        if not improved:
            step /= 2
            if step < min_step:
                break
    return best_w, float(best), float(base)


def write_weights(weights: dict[str, float], config_path: Path | None, out_path: Path) -> Path:
    data = {}
    if config_path is not None and config_path.exists():
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    fat = dict(data.get("fatigue") or {})
    fat["impairment_weights"] = {k: round(float(v), 6) for k, v in weights.items()}
    data["fatigue"] = fat
    FatigueConfig(**fat)  # validate before writing
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Refit fatigue impairment weights from pilot logs")
    ap.add_argument("--runs", required=True, help="directory of pilot run dirs (history.jsonl)")
    ap.add_argument("--config", default=None, help="config YAML whose fatigue: section is the starting point")
    ap.add_argument("--out", default=None, help="output YAML (default <config>.refit.yaml or fatigue_refit.yaml)")
    ap.add_argument("--in-place", action="store_true", help="overwrite --config")
    ap.add_argument("--arm", default=None, help="policy whose runs are used (default fatigue.validation.arm)")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=30)
    a = ap.parse_args(argv)

    cfg_path = Path(a.config) if a.config else None
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path else {}
    cfg = FatigueConfig(**((data or {}).get("fatigue") or {}))
    arm = cfg.validation.arm if a.arm is None else a.arm
    runs = [r for r in find_runs(a.runs) if not arm or r.policy == arm]
    if not runs:
        print(f"no runs for arm {arm!r} under {a.runs}", file=sys.stderr)
        return 2
    feats = [run_features(r, cfg, a.k) for r in runs]
    w, fitted, base = refit(feats, cfg, seed=a.seed, samples=a.samples, rounds=a.rounds)
    weights = dict(zip(COMPONENTS, w.tolist()))
    if a.in_place:
        if cfg_path is None:
            ap.error("--in-place needs --config")
        out = cfg_path
    elif a.out:
        out = Path(a.out)
    else:
        out = cfg_path.with_suffix(".refit.yaml") if cfg_path else Path("fatigue_refit.yaml")
    write_weights(weights, cfg_path, out)
    per_problem = problem_aucs(feats, cfg, w)
    print(json.dumps({"runs": len(runs), "arm": arm, "mean_auc_before": base, "mean_auc_after": fitted,
                      "weights": {k: round(v, 6) for k, v in weights.items()}, "written": str(out),
                      "per_problem": [p.to_dict() for p in per_problem]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
