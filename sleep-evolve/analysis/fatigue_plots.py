"""Fatigue over steps for every arm, plus a numeric shape check of the expected behavior.

Expected: fatigue climbs in the no-sleep arm, and drops after each sleep in the full-sleep arm.
`shape_report` checks that numerically (no plotting needed); `plot_fatigue` draws it. matplotlib
is imported lazily (optional `analysis` dependency group).

CLI:  uv run python -m analysis.fatigue_plots --runs runs --out fatigue_plots [--config cfg.yaml] [--logged]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.fatigue_logs import RunLog, fatigue_table, find_runs, load_fatigue_config
from harness.fatigue import FatigueConfig


def fatigue_series(run: RunLog, cfg: FatigueConfig | None = None, *, recompute: bool = True) -> dict:
    t = fatigue_table(run, cfg, recompute=recompute)
    return {"step": np.asarray(t["step"], float), "fatigue": np.asarray(t["fatigue"], float),
            "level": list(t["level"]),
            "sleep_steps": [s for s, trig in zip(t["step"], t["triggered"]) if trig],
            "depths": [d for d, trig in zip(t["depth"], t["triggered"]) if trig]}


def shape_report(runs: list[RunLog], cfg: FatigueConfig | None = None, *, recompute: bool = True) -> list[dict]:
    """Per run: OLS slope of fatigue vs step, number of sleeps, and the mean fatigue drop across a
    sleep (fatigue at the triggering step minus fatigue at the next step)."""
    out = []
    for r in runs:
        s = fatigue_series(r, cfg, recompute=recompute)
        x, y = s["step"], s["fatigue"]
        slope = float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 and np.ptp(x) > 0 else None
        idx = {st: i for i, st in enumerate(x.tolist())}
        drops = [float(y[idx[st]] - y[idx[st] + 1]) for st in s["sleep_steps"]
                 if st in idx and idx[st] + 1 < len(y)]
        out.append({"run_id": r.run_id, "problem_id": r.problem_id, "policy": r.policy, "seed": r.seed,
                    "slope": slope, "n_sleeps": len(s["sleep_steps"]),
                    "mean_drop_after_sleep": float(np.mean(drops)) if drops else None,
                    "final_fatigue": float(y[-1]) if len(y) else None})
    return out


def plot_fatigue(runs: list[RunLog], out_dir: str | Path, cfg: FatigueConfig | None = None, *,
                 recompute: bool = True) -> list[Path]:
    """One PNG per problem: fatigue vs step, one line per run, colored by arm; sleeps marked."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:  # optional dependency
        raise RuntimeError("plotting needs matplotlib (optional `analysis` dependency group)") from e
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_problem: dict = {}
    for r in runs:
        by_problem.setdefault(r.problem_id, []).append(r)
    palette = ["#2a6fdb", "#e0693b", "#2f9e6f", "#a35bd6", "#c9a227", "#5c6773"]
    paths = []
    for pid, rs in sorted(by_problem.items(), key=lambda kv: (kv[0] is None, kv[0] or 0)):
        arms = sorted({str(r.policy) for r in rs})
        color = {a: palette[i % len(palette)] for i, a in enumerate(arms)}
        fig, ax = plt.subplots(figsize=(8, 4.5))
        seen = set()
        for r in sorted(rs, key=lambda r: (str(r.policy), r.seed or 0)):
            s = fatigue_series(r, cfg, recompute=recompute)
            a = str(r.policy)
            ax.plot(s["step"], s["fatigue"], color=color[a], lw=1.6, alpha=0.85,
                    label=a if a not in seen else None)
            seen.add(a)
            if s["sleep_steps"]:
                ys = np.interp(s["sleep_steps"], s["step"], s["fatigue"])
                ax.scatter(s["sleep_steps"], ys, marker="v", s=28, color=color[a], zorder=3)
        if cfg is not None:
            for name, th in (("drowsy", cfg.drowsy_threshold), ("tired", cfg.tired_threshold),
                             ("exhausted", cfg.exhausted_threshold)):
                ax.axhline(th, color="#999999", lw=0.8, ls=":")
                ax.annotate(name, xy=(1.0, th), xycoords=("axes fraction", "data"), xytext=(3, 0),
                            textcoords="offset points", va="center", fontsize=8, color="#666666")
        ax.set_xlabel("agent step")
        ax.set_ylabel("fatigue")
        ax.set_title(f"Fatigue by arm, problem {pid} (triangles = sleeps)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
        p = out / f"fatigue_p{pid}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fatigue over steps per arm")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="fatigue_plots")
    ap.add_argument("--config", default=None)
    ap.add_argument("--logged", action="store_true", help="plot logged fatigue instead of recomputing")
    ap.add_argument("--no-plot", action="store_true", help="only print the shape report")
    a = ap.parse_args(argv)
    cfg = load_fatigue_config(a.config)
    runs = find_runs(a.runs)
    print(json.dumps(shape_report(runs, cfg, recompute=not a.logged), indent=2))
    if not a.no_plot:
        for p in plot_fatigue(runs, a.out, cfg, recompute=not a.logged):
            print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
