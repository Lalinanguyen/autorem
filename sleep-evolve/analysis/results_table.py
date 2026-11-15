"""Experiment inventory and results tables for the dream-ladder suite (reporting only).

Reads runs/dream_openai/p{problem}_{arm}_s{seed}/ artifacts and prints:

  1. an inventory of all expected problem x arm runs with a status classification
     (complete | incomplete | missing | invalid_artifact),
  2. the primary results table: best valid score per (problem, arm),
  3. a per-run detail table: best score, evaluations, tokens, cost, runtime,
  4. direction-aware relative improvement vs no_sleep (scores are already in
     higher-is-better space; minimization problems store negated quantities, so
     (treatment - baseline) / |baseline| is correct for both directions).

Usage:
    uv run python -m analysis.results_table --runs runs/dream_openai
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED_PROBLEMS = [7, 36, 50, 59]
EXPECTED_ARMS = ["no_sleep", "fixed_letta", "summarize", "dream_unvalidated", "dream"]

# Higher-is-better score space is what summary.json records (minimization problems
# store the negated quantity). Direction labels for the report only.
DIRECTION = {7: "minimize |B|^2/n", 36: "maximize sum radii", 50: "minimize (max/min)^2",
             59: "maximize subset size"}


def classify(run_dir: Path) -> tuple[str, dict | None]:
    """(status, summary). complete = summary.json parses and has a best_score or a
    recorded stop_reason; invalid_artifact = files unreadable."""
    if not run_dir.is_dir():
        return "missing", None
    if not (run_dir / "history.jsonl").is_file():
        return "invalid_artifact", None
    sp = run_dir / "summary.json"
    if not sp.is_file():
        return "incomplete", None
    try:
        summary = json.loads(sp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "invalid_artifact", None
    return "complete", summary


def tokens_used(s: dict) -> int | None:
    b = s.get("budget") or {}
    if b.get("agent_tokens") is None and b.get("sleep_tokens") is None:
        usage = s.get("usage") or {}
        total = usage.get("total") or {}
        return total.get("total_tokens")
    return (b.get("agent_tokens") or 0) + (b.get("sleep_tokens") or 0)


def evaluations(s: dict) -> int | None:
    e = s.get("evaluations") or {}
    return e.get("used") or e.get("total") or e.get("agent")


def cost_usd(s: dict) -> float | None:
    c = s.get("cost") or {}
    t = c.get("total")
    return float(t) if isinstance(t, (int, float)) else None


def rel_improvement(treatment: float | None, baseline: float | None) -> float | None:
    """Percent improvement over baseline in higher-is-better score space.

    Returns None when the comparison is undefined (missing scores, or |baseline| ~ 0,
    where a percentage would be misleading — inspect the absolute scores instead).
    """
    if treatment is None or baseline is None or abs(baseline) < 1e-12:
        return None
    return (treatment - baseline) / abs(baseline) * 100.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/dream_openai")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default="", help="also dump the parsed inventory to this path")
    args = ap.parse_args()
    root = Path(args.runs)

    rows: list[dict] = []
    print("=== INVENTORY ===")
    print(f"{'problem':>8} {'arm':<20} {'status':<17} {'best_score':>14} {'norm':>7} "
          f"{'evals':>6} {'tokens':>8} {'cost':>8} {'runtime_s':>9} {'stop_reason'}")
    for p in EXPECTED_PROBLEMS:
        for arm in EXPECTED_ARMS:
            d = root / f"p{p}_{arm}_s{args.seed}"
            status, s = classify(d)
            row = {"problem": p, "arm": arm, "status": status, "run_dir": str(d)}
            if s:
                row.update(
                    best_score=s.get("best_score"), best_normalized=s.get("best_normalized"),
                    best_known=s.get("best_known"), steps=s.get("steps_completed"),
                    evaluations=evaluations(s), tokens=tokens_used(s),
                    cost=cost_usd(s), wallclock_s=s.get("wallclock_s"),
                    stop_reason=s.get("stop_reason"), error=s.get("error"),
                    llm_model=s.get("llm_model"), config_model=s.get("config_model"),
                    usage=s.get("usage"), budget=s.get("budget"),
                    metrics=s.get("metrics"), fatigue=s.get("fatigue"),
                )
            rows.append(row)
            if status == "complete":
                print(f"{p:>8} {arm:<20} {status:<17} {row['best_score']!s:>14.14} "
                      f"{row['best_normalized'] or 0:>7.4f} {row['evaluations']!s:>6.6} "
                      f"{row['tokens']!s:>8.8} {row['cost']!s:>8.8} {row['wallclock_s']!s:>9.9} "
                      f"{row['stop_reason']}")
            else:
                print(f"{p:>8} {arm:<20} {status:<17}")

    done = [r for r in rows if r["status"] == "complete"]
    print(f"\n{len(done)}/{len(rows)} complete; "
          f"{sum(1 for r in rows if r['status'] == 'incomplete')} incomplete; "
          f"{sum(1 for r in rows if r['status'] == 'missing')} missing; "
          f"{sum(1 for r in rows if r['status'] == 'invalid_artifact')} invalid")

    by = {(r["problem"], r["arm"]): r for r in done}

    print("\n=== PRIMARY TABLE: best valid score (normalized vs best_known in parens) ===")
    header = "| Problem | " + " | ".join(EXPECTED_ARMS) + " |"
    print(header)
    print("|" + "---|" * (len(EXPECTED_ARMS) + 1))
    for p in EXPECTED_PROBLEMS:
        cells = []
        for arm in EXPECTED_ARMS:
            r = by.get((p, arm))
            if not r or r["best_score"] is None:
                cells.append("—")
            else:
                bs, bn = r["best_score"], r["best_normalized"]
                cells.append(f"{bs:.6g} ({bn:.3f})" if bn is not None else f"{bs:.6g}")
        print(f"| P{p} ({DIRECTION.get(p, '')}) | " + " | ".join(cells) + " |")

    print("\n=== RELATIVE IMPROVEMENT vs no_sleep (higher-is-better score space) ===")
    print("| Problem | arm | improvement |")
    print("|---|---|---:|")
    for p in EXPECTED_PROBLEMS:
        base = (by.get((p, "no_sleep")) or {}).get("best_score")
        for arm in EXPECTED_ARMS[1:]:
            t = (by.get((p, arm)) or {}).get("best_score")
            ri = rel_improvement(t, base)
            txt = "n/a" if ri is None else f"{ri:+.2f}%"
            if t is not None and base is not None and abs(base) < 1e-12:
                txt = f"abs diff {t - base:+.4g} (baseline ~0)"
            print(f"| P{p} | {arm} | {txt} |")

    print("\n=== PER-RUN DETAIL ===")
    print("| Problem | Arm | Best score | Evals | Tokens | Cost USD | Runtime s |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for p in EXPECTED_PROBLEMS:
        for arm in EXPECTED_ARMS:
            r = by.get((p, arm))
            if r:
                fmt = lambda v: "—" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))
                print(f"| P{p} | {arm} | {fmt(r['best_score'])} | {fmt(r['evaluations'])} | "
                      f"{fmt(r['tokens'])} | {fmt(r['cost'])} | {fmt(r['wallclock_s'])} |")

    total_cost = sum(r["cost"] for r in done if r["cost"])
    total_tok = sum(r["tokens"] for r in done if r["tokens"])
    total_evals = sum(r["evaluations"] for r in done if r["evaluations"])
    print(f"\nTotals over {len(done)} completed runs: cost ${total_cost:.4f}, "
          f"tokens {total_tok:,}, evaluations {total_evals:,}"
          + (f", cost/eval ${total_cost / total_evals:.5f}" if total_evals else ""))

    if args.json:
        out = [{k: v for k, v in r.items() if k not in ("metrics", "budget", "usage")} for r in rows]
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
