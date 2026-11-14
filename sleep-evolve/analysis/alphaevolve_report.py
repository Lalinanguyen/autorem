"""AlphaEvolve benchmark report over a runs directory: markdown + PNG figures.

    uv run python -m analysis.alphaevolve_report --runs runs --out reports/alphaevolve
    uv run --group analysis python -m analysis.alphaevolve_report --runs runs --out reports/alphaevolve
    (add --no-figures to skip matplotlib; --write-summaries to also persist the alphaevolve blocks)

Each run's `alphaevolve` block (benchmarks/scoring.py) is used if present and computed with the
current registry; otherwise it is recomputed in memory. Guardrails:
  - runs that are not scored (problem not in the registry, evaluator pending, instance mismatch,
    failed verification) are listed as excluded with the reason and never enter a table;
  - groups are keyed by (problem, instance): different instance parameters are never compared;
  - a "beat AlphaEvolve" is reported as verified only with the program hash, our score, the
    published scorer's score and its validity output; otherwise it is listed as unverified;
  - the header records the git commit, models and budgets of the runs.
matplotlib is optional (the `analysis` dependency group) and imported lazily.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.registry import Registry, default_registry
from benchmarks.scoring import THRESHOLDS, compute_block, enrich_summary, find_run_dirs, progress, read_history

ROOT = Path(__file__).resolve().parents[1]

COMPUTE_NOTE = (
    "> **Compute context.** AlphaEvolve ran with far larger compute and expert prompting; our runs "
    "use a small fixed token budget. AlphaEvolve's values are shown as a **reference ceiling, not a "
    "head-to-head comparison**. The head-to-head comparison is between our arms at matched budgets."
)


@dataclass
class Run:
    run_dir: Path
    summary: dict
    block: dict
    history: list[dict] = field(default_factory=list)

    @property
    def run_id(self) -> str:
        return str(self.summary.get("run_id") or self.run_dir.name)

    @property
    def arm(self) -> str:
        s = self.summary
        cfg = s.get("config") or {}
        return str(s.get("arm") or s.get("baseline_arm") or cfg.get("baseline_arm") or s.get("sleep_policy") or "?")

    @property
    def total_tokens(self) -> int | None:
        return (self.summary.get("budget") or {}).get("total_tokens")

    @property
    def key(self) -> tuple[int, str]:
        return (int(self.block["problem_id"]), str(self.block["instance_id"]))


def load_runs(runs_dir: str | Path, *, registry: Registry | None = None, write: bool = False,
              verify_beats: bool = False) -> list[Run]:
    reg = registry or default_registry()
    runs = []
    for d in find_run_dirs(runs_dir):
        if write:
            enrich_summary(d, registry=reg, verify_beats=verify_beats)
        summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        history = read_history(d / "history.jsonl")
        block = summary.get("alphaevolve")
        if not block or block.get("registry_sha256") != reg.sha256:
            block = compute_block(summary, history, run_dir=d, registry=reg, verify_beats=verify_beats)
        runs.append(Run(d, summary, block, history))
    return runs


# --------------------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------------------

def _median(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _fmt(x, nd=4) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, float)):
        if float(x).is_integer() and abs(x) >= 1:
            return f"{int(x)}"
        return f"{x:.{nd}g}" if abs(x) < 1e-3 else f"{x:.{nd + 2}g}"
    return str(x)


def group_runs(runs: list[Run]) -> tuple[dict[tuple[int, str], dict[str, list[Run]]], list[Run]]:
    groups: dict[tuple[int, str], dict[str, list[Run]]] = defaultdict(lambda: defaultdict(list))
    excluded = []
    for r in runs:
        if r.block.get("scored"):
            groups[r.key][r.arm].append(r)
        else:
            excluded.append(r)
    return groups, excluded


def arm_stats(runs: list[Run]) -> dict[str, Any]:
    b = [r.block for r in runs]
    reach = {}
    for x in THRESHOLDS:
        k = f"{x:g}"
        hits = [blk["budget_to_reach"][k] for blk in b if blk["budget_to_reach"][k]["reached"]]
        reach[k] = {"reached": len(hits), "median_fraction": _median([h["fraction"] for h in hits])}
    return {
        "n": len(runs),
        "median_normalized": _median([x["normalized_vs_ae"] for x in b]),
        "median_gap_closed": _median([x["gap_closed"] for x in b]),
        "gap_null": sum(1 for x in b if x["gap_closed"] is None),
        "matched": sum(1 for x in b if x["matched_ae"]),
        "beat": sum(1 for x in b if x["beat_ae"]),
        "beat_verified": sum(1 for x in b if x["beat_status"] == "verified"),
        "budgets": sorted({r.total_tokens for r in runs if r.total_tokens is not None}),
        "median_budget_at_best": _median([(x.get("budget_at_best") or {}).get("fraction") for x in b]),
        "reach": reach,
    }


def gap_curve(run: Run, grid: np.ndarray) -> np.ndarray | None:
    """gap_closed of the best-so-far score as a step function of the budget fraction."""
    blk = run.block
    seed, ref = blk.get("seed_score"), blk.get("alphaevolve_score")
    if seed is None or ref is None or ref - seed <= 0:
        return None
    prog = [p for p in progress(run.history, run.total_tokens) if p["fraction"] is not None]
    if not prog:
        return None
    out = np.full(grid.shape, np.nan)
    for p in prog:
        if p["best"] is None:
            continue
        out[grid >= p["fraction"]] = (p["best"] - seed) / (ref - seed)
    return out


# --------------------------------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------------------------------

def git_info() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=10, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True,
                               timeout=20).stdout.strip()
        return head + (" (working tree has uncommitted changes)" if dirty else "")
    except Exception:
        return "unavailable"


def render_markdown(runs: list[Run], registry: Registry, figures: list[Path] | None = None,
                    figures_note: str | None = None, out_dir: Path | None = None) -> str:
    groups, excluded = group_runs(runs)
    arms = sorted({a for g in groups.values() for a in g})
    scored = [r for g in groups.values() for rs in g.values() for r in rs]
    models = sorted({str(r.summary.get("llm_model") or r.summary.get("config_model") or "unknown") for r in runs})
    budgets = sorted({r.total_tokens for r in runs if r.total_tokens is not None})
    L: list[str] = ["# AlphaEvolve benchmark report", "", COMPUTE_NOTE, ""]
    L += [f"- git commit: `{git_info()}`",
          f"- registry: `benchmarks/alphaevolve.yaml` sha256 `{registry.sha256[:16]}` (retrieved {registry.retrieved})",
          f"- models: {', '.join(models) or '—'}",
          f"- token budgets (total per run): {', '.join(map(str, budgets)) or '—'}",
          f"- runs: {len(runs)} found, {len(scored)} scored, {len(excluded)} excluded", ""]

    # Main table
    L += ["## Main table", "",
          "One row per problem (and instance). Values are in the published units; per-arm cells are "
          "`median normalized_vs_ae / median gap_closed / matched+beat of n seeds`.", ""]
    head = ["problem", "instance", "status", "evaluator", "AlphaEvolve", "prior best", "current best"] + arms
    L += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for p in registry.problems:
        keys = sorted(k for k in groups if k[0] == p.problem_id)
        if not keys:
            inst = p.instances[0] if len(p.instances) == 1 else None
            vals = ([_fmt(inst.ae_value), _fmt(inst.prior_best_value.value), _fmt(inst.current_best_value.value)]
                    if inst else [f"{len(p.instances)} instances"] + ["—", "—"])
            L.append("| " + " | ".join([f"{p.problem_id}. {p.name}", inst.id if inst else "—", p.status,
                                        "pending" if p.evaluator_pending else "ready (no runs)", *vals]
                                       + ["no runs"] * len(arms)) + " |")
            continue
        for k in keys:
            inst = p.instance(k[1])
            cells = []
            for a in arms:
                rs = groups[k].get(a, [])
                if not rs:
                    cells.append("—")
                    continue
                s = arm_stats(rs)
                cells.append(f"{_fmt(s['median_normalized'])} / {_fmt(s['median_gap_closed'], 3)} / "
                             f"{s['matched']}+{s['beat']} of {s['n']}")
            L.append("| " + " | ".join([f"{p.problem_id}. {p.name}", inst.id, p.status, "verified",
                                        _fmt(inst.ae_value), _fmt(inst.prior_best_value.value),
                                        _fmt(inst.current_best_value.value), *cells]) + " |")
    L += ["", "`—` = missing in the sources (see the registry's missing_note) or no runs. gap_closed is null "
          "when the seed already meets the reference (nothing to close).", ""]

    # Status strata
    L += ["## By status category", "",
          "Expectation: `matched_optimal` problems should be reachable; `world_record` problems are hard.", "",
          "| status | problems | scored runs | median normalized_vs_ae | matched | beat (verified) |", "|---|---|---|---|---|---|"]
    for st in ("matched_optimal", "former_record", "worse_than_record", "world_record", "unlisted"):
        probs = [p.problem_id for p in registry.problems if p.status == st]
        rs = [r for r in scored if r.block["status"] == st]
        L.append(f"| {st} | {', '.join(map(str, probs)) or '—'} | {len(rs)} | "
                 f"{_fmt(_median([r.block['normalized_vs_ae'] for r in rs]))} | "
                 f"{sum(1 for r in rs if r.block['matched_ae'])} | "
                 f"{sum(1 for r in rs if r.block['beat_ae'])} ({sum(1 for r in rs if r.block['beat_status'] == 'verified')}) |")
    L.append("")

    # Head-to-head between arms
    L += ["## Head-to-head between arms (matched budgets)", ""]
    if not groups:
        L += ["No scored runs.", ""]
    for k in sorted(groups):
        p = registry.get(k[0])
        L += [f"### Problem {k[0]} ({p.name}), instance `{k[1]}`", ""]
        stats = {a: arm_stats(rs) for a, rs in groups[k].items()}
        all_budgets = {b for s in stats.values() for b in s["budgets"]}
        if len(all_budgets) > 1:
            L += [f"**Budgets differ across runs ({sorted(all_budgets)}): not a matched-budget comparison.**", ""]
        L += ["| arm | n | median normalized_vs_ae | median gap_closed | matched | beat | "
              + " | ".join(f"reached {x:g} (median budget frac)" for x in THRESHOLDS) + " | median budget frac at best |",
              "|" + "---|" * (7 + len(THRESHOLDS))]
        for a, s in sorted(stats.items(), key=lambda kv: -(kv[1]["median_normalized"] or -1e300)):
            reach = " | ".join(f"{s['reach'][f'{x:g}']['reached']}/{s['n']} ({_fmt(s['reach'][f'{x:g}']['median_fraction'], 3)})"
                               for x in THRESHOLDS)
            L.append(f"| {a} | {s['n']} | {_fmt(s['median_normalized'])} | {_fmt(s['median_gap_closed'], 3)} | "
                     f"{s['matched']} | {s['beat']} | {reach} | {_fmt(s['median_budget_at_best'], 3)} |")
        L.append("")
        tps = [t for t in (groups[k][next(iter(groups[k]))][0].block.get("third_party") or [])]
        if tps:
            L += ["Third-party references (same instance; published values, labeled third_party):", "",
                  "| system | value | evaluations / cost | " + " | ".join(f"{a}: median normalized" for a in sorted(groups[k])) + " | source |",
                  "|" + "---|" * (4 + len(groups[k]))]
            for i, t in enumerate(tps):
                if not t.get("compared"):
                    cells = ["—"] * len(groups[k])
                else:
                    cells = [_fmt(_median([r.block["third_party"][i].get("normalized_vs") for r in groups[k][a]]))
                             for a in sorted(groups[k])]
                L.append(f"| {t['system']} | {_fmt(t['value'], 10) if t['value'] is not None else 'missing'} | "
                         f"{t.get('evaluations_or_cost') or '—'} | {' | '.join(cells)} | {t['source']} |")
            L.append("")

    # Beat claims
    beats = [r for r in scored if r.block.get("beat_ae")]
    L += ["## Claims of beating AlphaEvolve", ""]
    if not beats:
        L += ["None.", ""]
    for r in beats:
        bv = r.block.get("beat_verification") or {}
        st = r.block.get("beat_status")
        complete = all(bv.get(k) is not None for k in ("program_sha256", "our_score", "published_score", "validity_output"))
        label = "VERIFIED" if (st == "verified" and complete) else ("REJECTED" if st == "rejected" else "UNVERIFIED")
        L += [f"- **{label}** `{r.run_id}` (arm {r.arm}): our best {r.block['our_best_score']!r} vs AlphaEvolve "
              f"{r.block['alphaevolve_score']!r}",
              f"  - program: `{r.run_dir / 'best_program.py'}` sha256 `{bv.get('program_sha256')}`",
              f"  - our evaluator: {bv.get('our_score')!r} (valid={bv.get('our_valid')}); published scorer "
              f"`{bv.get('published_scorer')}`: {bv.get('published_score')!r} (valid={bv.get('published_valid')})",
              f"  - validity output: {bv.get('validity_output')}; reason: {bv.get('reason')}"]
    L.append("")

    # Excluded
    L += ["## Excluded runs (never compared)", ""]
    if not excluded:
        L += ["None.", ""]
    else:
        L += ["| run | problem | arm | reason |", "|---|---|---|---|"]
        L += [f"| {r.run_id} | {r.block.get('problem_id')} | {r.arm} | {r.block.get('reason')} |" for r in excluded]
        L.append("")

    # Registry coverage
    L += ["## Registry coverage", "",
          "| problem | tier | status | evaluator | instances | missing values |", "|---|---|---|---|---|---|"]
    for p in registry.problems:
        missing = [f"{i.id}.{f}" for i in p.instances for f in ("alphaevolve_value", "prior_best_value", "current_best_value")
                   if getattr(i, f).value is None]
        L.append(f"| {p.problem_id} | {p.tier} | {p.status} | {p.evaluator} | {', '.join(i.id for i in p.instances)} | "
                 f"{', '.join(missing) or '—'} |")
    L.append("")

    L += ["## Figures", ""]
    if figures:
        for f in figures:
            rel = f.relative_to(out_dir) if out_dir else f
            L += [f"![{f.stem}]({rel.as_posix()})", ""]
    else:
        L += [figures_note or "No figures.", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------------------

def plot_figures(runs: list[Run], out_dir: str | Path, registry: Registry | None = None) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError("figures need matplotlib (optional `analysis` dependency group)") from e
    reg = registry or default_registry()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    groups, _ = group_runs(runs)
    paths: list[Path] = []
    if not groups:
        return paths
    keys = sorted(groups)
    arms = sorted({a for g in groups.values() for a in g})
    colors = {a: plt.cm.tab10(i % 10) for i, a in enumerate(arms)}
    grid = np.linspace(0, 1, 201)

    # 1. gap_closed vs budget, one line per arm (median over seeds), faceted by problem/instance
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4), squeeze=False)
    for ax, k in zip(axes[0], keys):
        for a in sorted(groups[k]):
            curves = [c for c in (gap_curve(r, grid) for r in groups[k][a]) if c is not None]
            if not curves:
                continue
            with np.errstate(all="ignore"):
                med = np.nanmedian(np.vstack(curves), axis=0) if len(curves) > 1 else curves[0]
            ax.step(grid, med, where="post", color=colors[a], label=f"{a} (n={len(curves)})")
        ax.axhline(1.0, color="gray", ls="--", lw=0.8)
        ax.set_title(f"p{k[0]} {k[1]}")
        ax.set_xlabel("budget fraction used")
        ax.set_ylabel("gap closed toward AlphaEvolve")
        ax.legend(fontsize=8)
    fig.tight_layout()
    p = out / "gap_closed_vs_budget.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)

    # 2. per problem: which arm got closest, and the budget fraction at which it got there
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4), squeeze=False)
    for ax, k in zip(axes[0], keys):
        names = sorted(groups[k])
        stats = [arm_stats(groups[k][a]) for a in names]
        vals = [s["median_normalized"] or 0.0 for s in stats]
        bars = ax.bar(range(len(names)), vals, color=[colors[a] for a in names])
        for b, s in zip(bars, stats):
            ax.annotate(f"budget {_fmt(s['median_budget_at_best'], 2)}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8)
        ax.axhline(1.0, color="gray", ls="--", lw=0.8)
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
        lo = min(vals) if vals else 0
        ax.set_ylim(max(0.0, lo - 0.05), max(1.02, max(vals) + 0.02) if vals else 1.02)
        ax.set_title(f"p{k[0]} {k[1]}: median normalized_vs_ae")
    fig.tight_layout()
    p = out / "closest_arm_per_problem.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths.append(p)
    return paths


def build_report(runs_dir: str | Path, out_dir: str | Path, *, figures: bool = True, write: bool = False,
                 verify_beats: bool = False, registry: Registry | None = None) -> Path:
    reg = registry or default_registry()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(runs_dir, registry=reg, write=write, verify_beats=verify_beats)
    figs, note = None, "Figures skipped (--no-figures)."
    if figures:
        try:
            figs = plot_figures(runs, out, reg)
            note = None if figs else "No scored runs to plot."
        except RuntimeError as e:
            note = f"Figures skipped: {e}."
    md = render_markdown(runs, reg, figs, note, out)
    path = out / "alphaevolve_report.md"
    path.write_text(md, encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="reports/alphaevolve")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--write-summaries", action="store_true", help="also persist alphaevolve blocks into summary.json")
    ap.add_argument("--verify-beats", action="store_true",
                    help="re-run beat candidates through the sandbox + published scorer when computing blocks")
    args = ap.parse_args(argv)
    path = build_report(args.runs, args.out, figures=not args.no_figures, write=args.write_summaries,
                        verify_beats=args.verify_beats)
    print(path)


if __name__ == "__main__":
    main()
