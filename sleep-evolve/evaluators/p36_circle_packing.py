"""Problem 36: pack n=26 disjoint circles in the unit square, maximizing the sum of radii.

Source: google-deepmind/alphaevolve_repository_of_problems,
experiments/packing_circles_max_sum_of_radii/packing_circles_max_sum_of_radii.ipynb
(problem page problems/36.html; paper arXiv:2511.02864, Section 6.19).

The notebook's check_construction requires centers of shape (n, 2), finite values,
radii of shape (n,) that are finite and >= 0, every circle inside [0,1]^2, and no pair
with r_i + r_j > dist(c_i, c_j). Its checks are exact. Here we allow a 1e-9 tolerance
on containment and overlap, as the task spec asks.

best_known is AlphaEvolve's Construction 1 for n=26 (previous SOTA 2.634), as computed
by np.sum(radii_26) over the notebook's data: 2.6359830849176067.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from evaluators.base import EvalResult, SubprocessEvaluator, invalid

N_CIRCLES = 26
TOL = 1e-9
BEST_KNOWN = 2.6359830849176067

_SEED = '''import numpy as np


def construct():
    """Baseline: 26 equal circles on a 6x5 grid. Returns (centers, radii)."""
    cols, rows, n = 6, 5, 26
    r = 0.5 / max(cols, rows)
    centers = []
    for k in range(n):
        i, j = divmod(k, cols)
        centers.append([(j + 0.5) / cols, (i + 0.5) / rows])
    return np.array(centers), np.full(n, r)
'''


def check_packing(centers: np.ndarray, radii: np.ndarray, n: int = N_CIRCLES, tol: float = TOL) -> str | None:
    """Return None if valid, otherwise a short reason. Mirrors the notebook's checks."""
    if centers.shape != (n, 2) or not np.isfinite(centers).all():
        return f"centers must be a finite array of shape ({n}, 2), got {centers.shape}"
    if radii.shape != (n,) or not np.isfinite(radii).all() or (radii < 0).any():
        return f"radii must be a finite, non-negative array of shape ({n},), got {radii.shape}"
    lo = radii[:, None] - centers  # > 0 means it crosses the left/bottom edge
    hi = centers + radii[:, None] - 1.0  # > 0 means it crosses the right/top edge
    worst_edge = float(max(lo.max(), hi.max()))
    if worst_edge > tol:
        return f"circle outside the unit square by {worst_edge:.3g}"
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.sqrt((diff**2).sum(-1))
    overlap = radii[:, None] + radii[None, :] - dist
    iu = np.triu_indices(n, k=1)
    worst = float(overlap[iu].max()) if n > 1 else -np.inf
    if worst > tol:
        return f"circles overlap by {worst:.3g}"
    return None


class CirclePacking26(SubprocessEvaluator):
    problem_id = 36
    name = "p36_circle_packing_sum_radii_n26"
    best_known = BEST_KNOWN
    description = (
        "Pack 26 disjoint circles inside the unit square [0,1]x[0,1] to maximize the sum of "
        "their radii. Write a Python program defining construct() that returns "
        "(centers, radii): centers is a (26, 2) array of (x, y) circle centers and radii is "
        "a (26,) array of non-negative radii. Every circle must lie inside the square "
        "(r <= x <= 1 - r and r <= y <= 1 - r) and no two circles may overlap "
        "(r_i + r_j <= |c_i - c_j|), up to a tolerance of 1e-9. Score = sum of radii "
        f"(best known: {BEST_KNOWN:.6f}). numpy and the standard library are available. "
        "construct() must finish within the time limit."
    )

    def __init__(self, timeout_s: int = 60):
        self.timeout_s = timeout_s

    def seed_program(self) -> str:
        return _SEED

    def score_solution(self, solution: Any) -> EvalResult:
        if isinstance(solution, dict):
            centers, radii = solution.get("centers"), solution.get("radii")
        elif isinstance(solution, list) and len(solution) >= 2:
            centers, radii = solution[0], solution[1]  # extra items (e.g. claimed score) ignored
        else:
            return invalid("construct() must return (centers, radii)")
        try:
            c = np.asarray(centers, dtype=float)
            r = np.asarray(radii, dtype=float)
        except (TypeError, ValueError) as e:
            return invalid(f"could not convert to arrays: {e}")
        reason = check_packing(c, r)
        if reason:
            return invalid(reason)
        total = float(r.sum())
        return EvalResult(score=total, valid=True, details={"sum_of_radii": total})
