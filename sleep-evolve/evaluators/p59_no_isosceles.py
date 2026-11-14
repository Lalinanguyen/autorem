"""Problem 59: largest subset of the n x n grid with no isosceles triangle (n=64).

Source: google-deepmind/alphaevolve_repository_of_problems,
experiments/subsets_of_the_grid_with_no_isosceles_triangles/....ipynb
(problem page problems/59.html; paper arXiv:2511.02864, Problem 6.59).

The notebook's verify_construction rejects a set iff some ordered triple of distinct
points (a, b, c) has dist(a, b) == dist(b, c): b is the apex of an isosceles triangle
and the degenerate "flat" case (three collinear points with the middle one equidistant)
is included automatically. Checking that each point's squared distances to the others
are all distinct is the same condition, in exact integer arithmetic.

This instance: n=64, 0-based coordinates {0..n-1}^2 (the page's [n]^2 is equivalent by
translation). best_known is AlphaEvolve's sol_64: 112 points (notebook cell 1).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from evaluators.base import EvalResult, SubprocessEvaluator, invalid

GRID = 64
BEST_KNOWN = 112.0  # |sol_64|
MAX_LISTED = 10  # violations listed per kind in side_info (counts are always complete)

_SEED = '''import numpy as np


def construct():
    """Greedy baseline: scan the grid row-major, add every point that keeps the set
    isosceles-free (an apex check in exact integer arithmetic). Deterministic."""
    n = 64
    pts = []
    seen = [set() for _ in range(n * n)]  # seen[i]: squared distances from point i to the set

    def ok(p):
        d2s = set()
        for q in pts:
            d2 = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2
            if d2 in d2s or d2 in seen[q[0] * n + q[1]]:
                return None
            d2s.add(d2)
        return d2s

    for x in range(n):
        for y in range(n):
            p = (x, y)
            d2s = ok(p)
            if d2s is not None:
                seen[p[0] * n + p[1]] = d2s
                for q in pts:
                    seen[q[0] * n + q[1]].add((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)
                pts.append(p)
    return [list(p) for p in pts]
'''


def _as_int_points(solution: Any) -> tuple[np.ndarray | None, str | None]:
    """Return (points as int64 array, error). Enforces grid points: finite, integer-valued."""
    raw = solution.get("points") if isinstance(solution, dict) else solution
    try:
        pts = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as e:
        return None, f"could not convert to an array: {e}"
    if pts.ndim != 2 or pts.shape[1] != 2:
        return None, f"points must be a (k, 2) array of (x, y) grid points, got shape {pts.shape}"
    if not np.isfinite(pts).all():
        return None, "non-finite coordinate"
    if not np.equal(pts, np.round(pts)).all():
        return None, "coordinates must be integer grid points"
    return pts.astype(np.int64), None


def find_violations(pts: np.ndarray) -> list[dict]:
    """Every apex whose squared distances to the other points are not all distinct.

    Returns {"apex", "points", "distance2"} entries (apex b with a, c such that
    d2(b, a) == d2(b, c)), matching the notebook's forbidden-triple condition.
    """
    out = []
    n = len(pts)
    for i in range(n):
        d2 = (pts[:, 0] - pts[i, 0]) ** 2 + (pts[:, 1] - pts[i, 1]) ** 2
        d2[i] = -1  # exclude self
        vals, counts = np.unique(d2, return_counts=True)
        for v, c in zip(vals[counts > 1], counts[counts > 1]):
            partners = [int(j) for j in np.nonzero(d2 == v)[0]]
            out.append({"apex": int(i), "points": partners[:3], "distance2": int(v)})
    return out


class NoIsoscelesGrid64(SubprocessEvaluator):
    problem_id = 59
    name = "p59_no_isosceles_grid64"
    best_known = BEST_KNOWN
    benchmark_instance = {"n": 64}
    description = (
        f"Find the largest possible subset of the {GRID}x{GRID} grid of points such that no "
        "three distinct selected points form an isosceles triangle: for every point p in the "
        "set, its distances to the other selected points must all be different (this also "
        "forbids three collinear points where the middle one is equidistant from the other "
        "two). Write a Python program defining construct() that returns a list of [x, y] "
        f"integer coordinate pairs, each with 0 <= x, y <= {GRID - 1}, all distinct. "
        f"Score = the number of points (best known: {int(BEST_KNOWN)}). numpy and the "
        "standard library are available. construct() must finish within the time limit."
    )

    def __init__(self, timeout_s: int = 60):
        self.timeout_s = timeout_s

    def seed_program(self) -> str:
        return _SEED

    def score_solution(self, solution: Any) -> EvalResult:
        pts, err = _as_int_points(solution)
        if pts is None:
            return invalid(
                err,
                side_info={"stage": "shape", "expected": f"list of [x, y] integers in [0, {GRID - 1}]",
                           "got_type": type(solution).__name__},
            )
        if len(pts) > GRID * GRID:
            return invalid(f"more points than the grid has cells ({len(pts)} > {GRID * GRID})",
                           side_info={"stage": "shape", "size": len(pts)})
        oob = np.nonzero(((pts < 0) | (pts >= GRID)).any(1))[0]
        dup = len(pts) - len({tuple(p) for p in pts})
        violations = find_violations(pts) if len(pts) >= 3 else []
        side_info = {
            "stage": "constraint",
            "size": int(len(pts)),
            "num_out_of_bounds": int(len(oob)),
            "out_of_bounds": sorted(int(i) for i in oob)[:MAX_LISTED],
            "num_duplicate_points": int(dup),
            "num_isosceles_apexes": len(violations),
            "isosceles_apexes": [
                {"apex": [int(v) for v in pts[v["apex"]]],
                 "points": [[int(c) for c in pts[j]] for j in v["points"]],
                 "distance2": v["distance2"]}
                for v in violations[:MAX_LISTED]
            ],
        }
        if len(oob) or dup or violations:
            reasons = []
            if len(oob):
                reasons.append(f"{len(oob)} point(s) outside [0, {GRID - 1}]^2")
            if dup:
                reasons.append(f"{dup} duplicate point(s)")
            if violations:
                reasons.append(f"{len(violations)} isosceles apex(es)")
            return invalid("; ".join(reasons), side_info=side_info)
        side_info["stage"] = "ok"
        return EvalResult(score=float(len(pts)), valid=True,
                          details={"size": int(len(pts))}, side_info=side_info)
