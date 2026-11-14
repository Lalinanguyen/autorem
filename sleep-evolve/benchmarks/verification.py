"""Verification: our evaluator must reproduce AlphaEvolve's published value before any benchmark run.

`verify_problem(pid)` loads the published construction (JSON data in tests/data, extracted from the
notebook; never notebook code), turns it into a program that returns that data, scores it with OUR
evaluator through the normal sandbox, and checks:
  1. the registry entry has an evaluator (not `pending`) and evaluators.REGISTRY has it,
  2. the evaluator's declared instance equals a registry instance (no instance mismatch),
  3. the evaluator's direction agrees with the registry (minimization => negative best_known),
  4. the construction is valid and its score equals alphaevolve_value (in score space) within
     the instance's verification_tolerance,
  5. if the instance has a published scorer (a rewrite of the notebook's verification function),
     the construction also passes it with the same value.

`reverify_beat` re-checks a claimed "beat AlphaEvolve": the candidate program runs ONLY through the
evaluator sandbox (evaluators.base.run_program), its returned data is scored by our evaluator and by
the published scorer port, and the validity output of both is recorded. Without all three the beat
is `unverified`.

Published scorers are ports, rewritten here as data-driven code with the notebook's exact
(no-tolerance) semantics; the notebook itself is never executed.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from benchmarks.evaluator_instances import declared_instance
from benchmarks.registry import Instance, Problem, Registry, default_registry

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------------------------------
# Published scoring code (ports)
# --------------------------------------------------------------------------------------------------

def _p36_arrays(solution: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(solution, dict):
        centers, radii = solution.get("centers"), solution.get("radii")
    elif isinstance(solution, (list, tuple)) and len(solution) >= 2:
        centers, radii = solution[0], solution[1]
    else:
        raise ValueError("construction must be (centers, radii) or {'centers', 'radii'}")
    return np.asarray(centers, dtype=float), np.asarray(radii, dtype=float)


def p36_check_construction(solution: Any, params: dict[str, Any]) -> dict[str, Any]:
    """Port of check_construction (packing_circles_max_sum_of_radii.ipynb, cell 2). Exact checks.

    Same order and semantics as the notebook: centers shape (n, 2) and finite; every circle
    contained (r <= x <= 1 - r and r <= y <= 1 - r, no tolerance); radii shape (n,), finite, >= 0;
    no pair with r_i + r_j > dist(c_i, c_j). Returns the notebook's sum_of_radii (np.sum).
    """
    n = int(params["n"])
    if params.get("container") != "unit_square":
        return {"valid": False, "value": None, "messages": [f"unsupported container {params.get('container')!r}"]}
    try:
        centers, radii = _p36_arrays(solution)
    except (TypeError, ValueError) as e:
        return {"valid": False, "value": None, "messages": [f"bad construction: {e}"]}
    msgs: list[str] = []
    if centers.shape != (n, 2) or not np.isfinite(centers).all():
        return {"valid": False, "value": None,
                "messages": ["Error: The 'centers' array has an invalid shape or non-finite values."]}
    if radii.ndim != 1 or radii.shape[0] != n:
        return {"valid": False, "value": None, "messages": ["radii bad shape"]}
    is_contained = ((radii[:, None] <= centers) & (centers <= 1 - radii[:, None])).all(axis=1)
    if not is_contained.all():
        bad = [int(i) for i in np.nonzero(~is_contained)[0]]
        return {"valid": False, "value": None, "messages": [f"circles not contained: {bad[:10]}"]}
    if not np.isfinite(radii).all() or not (0 <= radii).all():
        return {"valid": False, "value": None, "messages": ["radii bad shape"]}
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if radii[i] + radii[j] > dist:
                return {"valid": False, "value": None, "messages": [f"circles overlap ({i}, {j})"]}
    msgs.append("The circles are disjoint and lie inside the unit square.")
    return {"valid": True, "value": float(np.sum(radii)), "messages": msgs}


PUBLISHED_SCORERS: dict[str, Callable[[Any, dict[str, Any]], dict[str, Any]]] = {
    "p36_check_construction": p36_check_construction,
}


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------

def program_returning(data: Any) -> str:
    """A candidate program whose construct() returns `data` (JSON round-trip, no code from sources)."""
    blob = json.dumps(json.dumps(data))
    return f"import json\n\n\ndef construct():\n    return json.loads({blob})\n"


def load_construction(instance: Instance) -> Any:
    if not instance.construction:
        raise FileNotFoundError(f"instance {instance.id} has no construction file")
    return json.loads((ROOT / instance.construction).read_text(encoding="utf-8"))


def evaluator_for(problem: Problem, **kwargs):
    """Our evaluator for a registry problem, or None if evaluators.REGISTRY has none."""
    from evaluators import REGISTRY, make_evaluator

    if problem.problem_id not in REGISTRY:
        return None
    return make_evaluator(problem.problem_id, **kwargs)


def resolve_instance(problem: Problem, evaluator) -> tuple[Instance | None, dict | None, str | None]:
    """(registry instance, evaluator's declared params, reason-if-refused)."""
    declared = declared_instance(evaluator) if evaluator is not None else None
    if declared is None:
        return None, None, "evaluator declares no instance (add benchmark_instance or benchmarks/evaluator_instances.py)"
    inst = problem.match_instance(declared)
    if inst is None:
        known = {i.id: i.params for i in problem.instances}
        return None, declared, f"instance mismatch: evaluator scores {declared}, registry has {known}"
    return inst, declared, None


# --------------------------------------------------------------------------------------------------
# Verification of the published construction
# --------------------------------------------------------------------------------------------------

_CACHE: dict[tuple[int, str], dict] = {}


def verify_problem(problem_id: int, *, registry: Registry | None = None, evaluator=None,
                   use_cache: bool = True) -> dict[str, Any]:
    """Reproduce alphaevolve_value with our evaluator. status: passed | failed | skipped."""
    reg = registry or default_registry()
    key = (problem_id, reg.sha256)
    if use_cache and evaluator is None and key in _CACHE:
        return _CACHE[key]
    out = _verify(problem_id, reg, evaluator)
    if use_cache and evaluator is None:
        _CACHE[key] = out
    return out


def _verify(problem_id: int, reg: Registry, evaluator) -> dict[str, Any]:
    problem = reg.get(problem_id)
    if problem is None:
        return {"status": "skipped", "problem_id": problem_id, "reason": "not in the benchmark registry"}
    ev = evaluator if evaluator is not None else evaluator_for(problem)
    if ev is None:
        return {"status": "skipped", "problem_id": problem_id,
                "reason": f"evaluator: {problem.evaluator} (no evaluator in evaluators.REGISTRY)"}
    res: dict[str, Any] = {"problem_id": problem_id, "evaluator": f"{type(ev).__module__}:{type(ev).__name__}"}
    if problem.evaluator_pending:
        return {**res, "status": "failed",
                "reason": "an evaluator is registered but the benchmark entry is `evaluator: pending`: add the "
                          "construction to tests/data, set `evaluator:` and `construction:` in benchmarks/alphaevolve.yaml"}
    inst, declared, why = resolve_instance(problem, ev)
    res["evaluator_instance"] = declared
    if inst is None:
        return {**res, "status": "failed", "reason": why}
    res["instance_id"] = inst.id
    if inst.ae_value is None:
        return {**res, "status": "failed", "reason": "alphaevolve_value is missing for this instance"}
    bk = float(getattr(ev, "best_known", float("nan")))
    if math.isfinite(bk) and bk != 0 and (bk < 0) != (inst.direction == "minimize"):
        return {**res, "status": "failed",
                "reason": f"direction mismatch: registry says {inst.direction}, evaluator best_known={bk}"}
    try:
        data = load_construction(inst)
    except FileNotFoundError as e:
        return {**res, "status": "failed", "reason": str(e)}
    result = ev.evaluate(program_returning(data))
    expected = inst.ae_score
    res.update(expected_score=expected, construction_score=result.score if result.valid else None,
               construction_valid=result.valid, tolerance=inst.verification_tolerance)
    if not result.valid:
        return {**res, "status": "failed", "reason": f"construction invalid under our evaluator: {result.error}"}
    if abs(result.score - expected) > inst.verification_tolerance:
        return {**res, "status": "failed",
                "reason": f"construction scores {result.score!r}, expected {expected!r} "
                          f"(|diff| {abs(result.score - expected):.3g} > {inst.verification_tolerance})"}
    if inst.published_scorer:
        pub = PUBLISHED_SCORERS[inst.published_scorer](data, inst.params)
        res["published"] = pub
        if not pub["valid"] or abs(inst.to_score(pub["value"]) - expected) > inst.verification_tolerance:
            return {**res, "status": "failed", "reason": f"published scorer disagrees: {pub}"}
    return {**res, "status": "passed", "reason": None}


# --------------------------------------------------------------------------------------------------
# Re-verification of a "beat AlphaEvolve" claim
# --------------------------------------------------------------------------------------------------

def reverify_beat(problem: Problem, instance: Instance, program: str | None, evaluator) -> dict[str, Any]:
    """Score a beat candidate with our evaluator AND the published scorer port.

    beat_status: verified (both valid, both above alphaevolve_value + match_tolerance) | rejected
    (published scorer or our evaluator says invalid / not above) | unverified (anything missing).
    """
    out: dict[str, Any] = {"beat_status": "unverified", "program_sha256": None, "our_score": None,
                           "our_valid": None, "our_error": None, "published_scorer": instance.published_scorer,
                           "published_value": None, "published_score": None, "published_valid": None,
                           "validity_output": None, "reason": None}
    if not program:
        out["reason"] = "no program available to re-verify"
        return out
    out["program_sha256"] = hashlib.sha256(program.encode("utf-8")).hexdigest()
    if not instance.published_scorer or instance.published_scorer not in PUBLISHED_SCORERS:
        out["reason"] = "no port of the published scoring code for this instance"
        return out
    from evaluators.base import ProgramFailed, run_program

    try:
        solution = run_program(program, getattr(evaluator, "timeout_s", 60))  # sandbox only
    except ProgramFailed as e:
        out.update(beat_status="rejected", reason=f"program failed in the sandbox: {e}")
        return out
    score_solution = getattr(evaluator, "score_solution", None)
    ours = score_solution(solution) if callable(score_solution) else evaluator.evaluate(program)
    out.update(our_score=ours.score if ours.valid else None, our_valid=ours.valid, our_error=ours.error)
    pub = PUBLISHED_SCORERS[instance.published_scorer](solution, instance.params)
    out.update(published_value=pub["value"], published_valid=pub["valid"],
               published_score=instance.to_score(pub["value"]), validity_output=pub["messages"])
    threshold = instance.ae_score + instance.match_tolerance
    if not ours.valid or not pub["valid"]:
        out.update(beat_status="rejected", reason="invalid under " + ("our evaluator" if not ours.valid else "the published scorer"))
    elif ours.score > threshold and out["published_score"] > threshold:
        out.update(beat_status="verified", reason=None)
    else:
        out.update(beat_status="rejected", reason="not above alphaevolve_value + tolerance under both scorers")
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Verify every benchmark problem that has an evaluator.")
    ap.parse_args()
    failed = False
    for p in default_registry().problems:
        r = verify_problem(p.problem_id)
        failed |= r["status"] == "failed"
        print(f"p{p.problem_id:<3} {r['status']:<8} {r.get('instance_id') or '':<18} {r.get('reason') or ''}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
