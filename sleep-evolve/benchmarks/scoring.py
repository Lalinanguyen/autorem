"""Score a finished run against AlphaEvolve: adds an `alphaevolve` block to summary.json.

    uv run python -m benchmarks.scoring runs                 # every run dir under runs/
    uv run python -m benchmarks.scoring runs/p36_gepa_s0     # one run
    uv run python -m benchmarks.scoring runs --no-verify-beats

`enrich_summary(run_dir)` reads summary.json + history.jsonl, computes the block, re-reads
summary.json and rewrites it with only the `alphaevolve` key replaced (other keys, e.g. the
baseline suite's `metrics` block, are left untouched). Running it twice gives the same file.

All comparisons happen in our higher-is-better SCORE space (score = value for maximization,
score = -value for minimization, as the evaluators do), so one formula serves both directions:
  normalized_vs_ae = evaluators.base.normalized_score(our_best, ae_score)
                     (our/AE for maximize, AE/our for minimize; 1.0 = matches AlphaEvolve)
  gap_closed       = (our_best - seed) / (ae_score - seed); null with gap_closed_reason
                     "seed_at_or_above_reference" when ae_score - seed <= 0 (nothing to close),
                     "seed_invalid" / "no_valid_program" when a score is missing
  matched_ae       = our_best >= ae_score - match_tolerance
  beat_ae          = our_best >  ae_score + match_tolerance; then re-verified with the published
                     scoring code (benchmarks/verification.py). beat_status: not_claimed |
                     verified | rejected | unverified
  budget_to_reach  for X in 0.9/0.95/0.99: first step whose best-so-far normalized_vs_ae >= X,
                   with fraction = cumulative (agent + sleep) tokens at that step / total budget;
                   reached: false (fraction null) if never
The run is refused (scored: false, with a reason) when the problem is not in the registry, its
evaluator is pending, the evaluator's instance differs from every registry instance, or the
published construction does not reproduce alphaevolve_value with our evaluator.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from benchmarks.registry import Instance, Problem, Registry, default_registry

SCHEMA_VERSION = 1
THRESHOLDS = (0.9, 0.95, 0.99)


# --------------------------------------------------------------------------------------------------
# Pure comparison functions (score space, higher is better)
# --------------------------------------------------------------------------------------------------

def _finite(x: Any) -> float | None:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def normalized_vs(our: float | None, ref_score: float | None) -> float | None:
    from evaluators.base import normalized_score

    our, ref_score = _finite(our), _finite(ref_score)
    if our is None or ref_score is None:
        return None
    return normalized_score(our, ref_score)


def gap_closed(our: float | None, seed: float | None, ref_score: float | None) -> tuple[float | None, str | None]:
    """(gap_closed, reason-if-null). Fraction of the seed->reference distance covered."""
    our, seed, ref_score = _finite(our), _finite(seed), _finite(ref_score)
    if ref_score is None:
        return None, "reference_missing"
    if seed is None:
        return None, "seed_invalid"
    if our is None:
        return None, "no_valid_program"
    denom = ref_score - seed
    if denom <= 0:
        return None, "seed_at_or_above_reference"
    return (our - seed) / denom, None


def compare(our: float | None, seed: float | None, ref_score: float | None, tol: float) -> dict[str, Any]:
    g, why = gap_closed(our, seed, ref_score)
    o = _finite(our)
    r = _finite(ref_score)
    return {
        "normalized": normalized_vs(o, r),
        "gap_closed": g,
        "gap_closed_reason": why,
        "matched": (o >= r - tol) if (o is not None and r is not None) else None,
        "beat": (o > r + tol) if (o is not None and r is not None) else None,
    }


def step_records(history: list[dict]) -> list[dict]:
    return [h for h in history if h.get("event", "step") == "step"]


def progress(history: list[dict], total_tokens: int | None) -> list[dict]:
    """Best-so-far score per step line with its cumulative token use and budget fraction."""
    out, best = [], None
    for h in step_records(history):
        s = _finite(h.get("score")) if h.get("valid", True) else None
        if s is not None and (best is None or s > best):
            best = s
        tokens = int(h.get("agent_tokens") or 0) + int(h.get("sleep_tokens") or 0)
        out.append({"step": h.get("step"), "tokens": tokens,
                    "fraction": (tokens / total_tokens) if total_tokens else None, "best": best})
    return out


def budget_to_reach(prog: list[dict], ref_score: float | None, thresholds=THRESHOLDS) -> dict[str, dict]:
    res = {}
    for x in thresholds:
        hit = None
        for p in prog:
            n = normalized_vs(p["best"], ref_score)
            if n is not None and n >= x:
                hit = p
                break
        res[f"{x:g}"] = ({"reached": True, "step": hit["step"], "tokens": hit["tokens"], "fraction": hit["fraction"]}
                         if hit else {"reached": False, "step": None, "tokens": None, "fraction": None})
    return res


def budget_at_best(prog: list[dict]) -> dict | None:
    final = prog[-1]["best"] if prog else None
    if final is None:
        return None
    for p in prog:
        if p["best"] is not None and p["best"] >= final:
            return {"step": p["step"], "tokens": p["tokens"], "fraction": p["fraction"]}
    return None


# --------------------------------------------------------------------------------------------------
# The alphaevolve block
# --------------------------------------------------------------------------------------------------

def _seed_score(history: list[dict]) -> float | None:
    for h in step_records(history):
        if h.get("step") == 0:
            return _finite(h.get("score")) if h.get("valid") else None
    return None


def _value_record(rec) -> dict:
    return {"value": rec.value, "source": rec.source, "missing_note": rec.missing_note}


def compute_block(summary: dict, history: list[dict], *, run_dir: str | Path | None = None,
                  registry: Registry | None = None, evaluator=None, verify: bool = True,
                  verify_beats: bool = True) -> dict[str, Any]:
    reg = registry or default_registry()
    pid = summary.get("problem_id")
    base = {"schema_version": SCHEMA_VERSION, "registry_sha256": reg.sha256, "problem_id": pid}
    problem = reg.get(pid) if isinstance(pid, int) else None
    if problem is None:
        return {**base, "in_benchmark": False, "scored": False,
                "reason": f"problem {pid} is not in the AlphaEvolve benchmark registry (benchmarks/alphaevolve.yaml)"}
    base.update(in_benchmark=True, name=problem.name, tier=problem.tier, status=problem.status,
                status_raw=problem.status_raw, evaluator=problem.evaluator)
    if problem.evaluator_pending:
        return {**base, "scored": False,
                "reason": "evaluator: pending in the registry; not scored or reported as verified until an "
                          "evaluator exists and reproduces AlphaEvolve's construction"}

    from benchmarks.verification import evaluator_for, resolve_instance, reverify_beat, verify_problem

    ev = evaluator if evaluator is not None else evaluator_for(problem)
    if ev is None:
        return {**base, "scored": False, "reason": f"no evaluator in evaluators.REGISTRY for problem {pid}"}
    inst, declared, why = resolve_instance(problem, ev)
    base["evaluator_instance"] = declared
    if inst is None:
        return {**base, "scored": False, "instance_match": False, "reason": why}
    base.update(instance_match=True, instance_id=inst.id, instance_params=inst.params,
                direction=inst.direction, score_transform=inst.score_transform)
    if verify:
        v = verify_problem(pid, registry=reg, evaluator=evaluator)
        base["verification"] = {"status": v["status"], "reason": v.get("reason"),
                                "construction_score": v.get("construction_score"),
                                "expected_score": v.get("expected_score"), "tolerance": v.get("tolerance")}
        if v["status"] != "passed":
            return {**base, "scored": False, "reason": f"verification {v['status']}: {v.get('reason')}"}
    else:
        base["verification"] = {"status": "not_run"}
    if inst.ae_value is None:
        return {**base, "scored": False, "reason": f"alphaevolve_value missing: {inst.alphaevolve_value.missing_note}"}

    total = (summary.get("budget") or {}).get("total_tokens")
    prog = progress(history, total)
    seed = _seed_score(history)
    our = _finite(summary.get("best_score"))
    if our is None and prog:
        our = prog[-1]["best"]
    tol = inst.match_tolerance
    c = compare(our, seed, inst.ae_score, tol)

    block = {
        **base, "scored": True, "reason": None,
        "alphaevolve_value": inst.ae_value, "alphaevolve_score": inst.ae_score,
        "alphaevolve_source": inst.alphaevolve_value.source,
        "prior_best_value": _value_record(inst.prior_best_value),
        "current_best_value": {**_value_record(inst.current_best_value), "holder": inst.current_best_value.holder},
        "seed_score": seed, "our_best_score": our, "our_best_value": inst.from_score(our),
        "normalized_vs_ae": c["normalized"], "gap_closed": c["gap_closed"],
        "gap_closed_reason": c["gap_closed_reason"],
        "matched_ae": c["matched"], "beat_ae": c["beat"], "match_tolerance": tol,
        "beat_status": "not_claimed", "beat_verification": None,
        "total_budget_tokens": total,
        "budget_to_reach": budget_to_reach(prog, inst.ae_score),
        "budget_at_best": budget_at_best(prog),
        "third_party": [],
    }
    if c["beat"]:
        program = _best_program(run_dir, summary, history)
        if verify_beats:
            bv = reverify_beat(problem, inst, program, ev)
        else:
            bv = {"beat_status": "unverified", "reason": "re-verification not run (--no-verify-beats)"}
        block["beat_status"] = bv["beat_status"]
        block["beat_verification"] = bv
    for tp in inst.third_party:
        entry = {"system": tp.system, "value": tp.value, "comparable": tp.comparable, "source": tp.source,
                 "evaluations_or_cost": tp.evaluations_or_cost}
        if tp.value is None:
            entry.update(compared=False, reason=f"missing: {tp.missing_note}")
        elif not tp.comparable:
            entry.update(compared=False, reason="different instance; not compared")
        else:
            tc = compare(our, seed, inst.to_score(tp.value), tol)
            entry.update(compared=True, normalized_vs=tc["normalized"], gap_closed=tc["gap_closed"],
                         gap_closed_reason=tc["gap_closed_reason"], matched=tc["matched"], beat=tc["beat"])
        block["third_party"].append(entry)
    return block


def _best_program(run_dir, summary: dict, history: list[dict]) -> str | None:
    if run_dir is not None:
        p = Path(run_dir) / "best_program.py"
        if p.exists():
            return p.read_text(encoding="utf-8")
    bid = summary.get("best_program_id")
    for h in step_records(history):
        if bid is not None and h.get("program_id") == bid and h.get("program"):
            return h["program"]
    return None


# --------------------------------------------------------------------------------------------------
# File I/O
# --------------------------------------------------------------------------------------------------

def read_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _atomic_write_json(path: Path, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(prefix=".summary_", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def enrich_summary(run_dir: str | Path, *, registry: Registry | None = None, evaluator=None,
                   verify: bool = True, verify_beats: bool = True) -> dict:
    """Add/replace the `alphaevolve` block in run_dir/summary.json. Idempotent. Returns the block."""
    run_dir = Path(run_dir)
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    history = read_history(run_dir / "history.jsonl")
    block = compute_block(summary, history, run_dir=run_dir, registry=registry, evaluator=evaluator,
                          verify=verify, verify_beats=verify_beats)
    fresh = json.loads(summary_path.read_text(encoding="utf-8"))  # keep keys written meanwhile
    if fresh.get("alphaevolve") != block:
        fresh["alphaevolve"] = block
        _atomic_write_json(summary_path, fresh)
    return block


def enrich_after_run(summary: dict, out_dir: str | Path) -> dict:
    """Post-run hook for scripts/run_one.py and arms/run.py: enrich out_dir/<run_id>/summary.json.

    Never fails a finished run: an error is reported as a warning and the summary returned as is.
    Returns the summary dict with the `alphaevolve` block added.
    """
    import warnings

    try:
        block = enrich_summary(Path(out_dir) / str(summary["run_id"]))
    except Exception as e:  # the run's own outputs are already on disk
        warnings.warn(f"AlphaEvolve scoring failed for {summary.get('run_id')}: {type(e).__name__}: {e}")
        return summary
    return {**summary, "alphaevolve": block}


def find_run_dirs(path: str | Path) -> list[Path]:
    p = Path(path)
    if (p / "summary.json").exists():
        return [p]
    return sorted(d.parent for d in p.glob("*/summary.json"))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="a run dir (with summary.json) or a dir of run dirs")
    ap.add_argument("--no-verify-beats", action="store_true", help="mark beat claims unverified instead of re-running them")
    args = ap.parse_args(argv)
    dirs = find_run_dirs(args.path)
    if not dirs:
        sys.exit(f"no summary.json under {args.path}")
    for d in dirs:
        b = enrich_summary(d, verify_beats=not args.no_verify_beats)
        if b.get("scored"):
            print(f"{d.name}: normalized_vs_ae={b['normalized_vs_ae']} gap_closed={b['gap_closed']} "
                  f"matched={b['matched_ae']} beat={b['beat_ae']} ({b['beat_status']})")
        else:
            print(f"{d.name}: not scored: {b['reason']}")


if __name__ == "__main__":
    main()
