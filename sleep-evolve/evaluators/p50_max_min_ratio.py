"""Problem 50: place n points in d dimensions minimizing (max pairwise distance / min)^2.

Source: google-deepmind/alphaevolve_results, mathematical_results.ipynb section B.8
(problem page problems/50.html; paper arXiv:2511.02864, Problem 6.50).

This instance: d=2, n=16. The ratio is scale-invariant, so no bounding box: the only
validity requirements are the right shape, finite coordinates, and a nonzero minimum
pairwise distance (coincident points make the ratio infinite).

Registry direction is minimize, so the higher-is-better score is -(max/min)^2.
best_known is AlphaEvolve's construction_1 (notebook cell 93, verified by cell 94):
ratio^2 = 12.88926611203463.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from evaluators.base import EvalResult, SubprocessEvaluator, invalid

N_POINTS = 16
DIM = 2
BEST_RATIO_SQ = 12.88926611203463
BEST_KNOWN = -BEST_RATIO_SQ  # minimization: score is the negated ratio^2

_SEED = '''import numpy as np


def construct():
    """Baseline: 16 points on a 4x4 grid (ratio^2 = 18)."""
    g = np.linspace(0.0, 1.0, 4)
    return np.array([[x, y] for x in g for y in g])
'''


def _sig(x: float) -> float:
    """Round to 4 significant digits."""
    return float(f"{float(x):.4g}")


def ratio_side_info(points: np.ndarray) -> dict:
    """Diagnostics for a well-shaped point set: which pairs are tight, which points slack.

    nn[i] is point i's distance to its nearest neighbour; max_nn is the bottleneck the
    agent must raise, and the min/max pairs are the two constraints binding the ratio.
    """
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt((diff**2).sum(-1))
    np.fill_diagonal(dist, np.inf)
    n = len(points)
    iu, ju = np.triu_indices(n, k=1)
    pair = dist[iu, ju]
    k_min = int(pair.argmin())
    k_max = int(pair.argmax())
    nn = dist.min(1)
    return {
        "n": n,
        "min_distance": _sig(pair[k_min]),
        "max_distance": _sig(pair[k_max]),
        "ratio_squared": _sig((pair[k_max] / pair[k_min]) ** 2) if pair[k_min] > 0 else None,
        "min_pair": [int(iu[k_min]), int(ju[k_min])],
        "max_pair": [int(iu[k_max]), int(ju[k_max])],
        "nearest_neighbour_distances": sorted(_sig(v) for v in nn),
        "points_at_min_distance": sorted(
            int(i) for i in np.nonzero(nn <= pair[k_min] * (1 + 1e-9))[0]
        ),
    }


class MaxMinRatio16(SubprocessEvaluator):
    problem_id = 50
    name = "p50_max_min_ratio_d2_n16"
    best_known = BEST_KNOWN
    benchmark_instance = {"d": 2, "n": 16}
    description = (
        f"Place {N_POINTS} points in the plane minimizing (max pairwise distance / min pairwise "
        "distance)^2. Write a Python program defining construct() that returns a "
        f"({N_POINTS}, {DIM}) array of point coordinates; scale and position are free "
        "(the ratio is invariant under translation, rotation and scaling). All points must "
        "be finite and the minimum pairwise distance must be positive. "
        f"Score = -(ratio^2); best known ratio^2 is {BEST_RATIO_SQ:.9f} "
        f"(score {BEST_KNOWN:.9f}). numpy and the standard library are available. "
        "construct() must finish within the time limit."
    )

    def __init__(self, timeout_s: int = 60):
        self.timeout_s = timeout_s

    def seed_program(self) -> str:
        return _SEED

    def score_solution(self, solution: Any) -> EvalResult:
        raw = solution.get("points") if isinstance(solution, dict) else solution
        expected = {"points": [N_POINTS, DIM]}
        try:
            pts = np.asarray(raw, dtype=float)
        except (TypeError, ValueError) as e:
            return invalid(
                f"could not convert to an array: {e}",
                side_info={"stage": "shape", "expected": expected,
                           "got_type": type(solution).__name__, "message": str(e)[:300]},
            )
        if pts.shape != (N_POINTS, DIM) or not np.isfinite(pts).all():
            info = {"stage": "shape", "expected": expected, "got": list(pts.shape)}
            if pts.shape == (N_POINTS, DIM):
                info["non_finite_rows"] = [int(i) for i in np.nonzero(~np.isfinite(pts).all(1))[0]]
            return invalid(
                f"points must be a finite ({N_POINTS}, {DIM}) array, got shape {pts.shape}",
                side_info=info,
            )
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt((diff**2).sum(-1))
        iu, ju = np.triu_indices(N_POINTS, k=1)
        pair = dist[iu, ju]
        k_min = int(pair.argmin())
        if pair[k_min] <= 0:
            return invalid(
                "two points coincide (min pairwise distance is 0)",
                side_info={"stage": "constraint", "coincident_pair": [int(iu[k_min]), int(ju[k_min])]},
            )
        side_info = ratio_side_info(pts)
        ratio_sq = float((pair.max() / pair.min()) ** 2)
        return EvalResult(
            score=-ratio_sq, valid=True,
            details={"ratio_squared": ratio_sq, "min_distance": float(pair.min()),
                     "max_distance": float(pair.max())},
            side_info=side_info,
        )
