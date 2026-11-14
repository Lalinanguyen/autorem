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


MAX_LISTED = 10  # violations listed per kind in side_info (the counts are always complete)
TIGHT = 1e-6  # a constraint with slack below this counts as a tight contact
def _sig(x: float) -> float:
    """Round to 4 significant digits: enough to act on, compact in prompts and logs."""
    return float(f"{float(x):.4g}")


_GLYPHS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def render_packing(centers: np.ndarray, radii: np.ndarray, cols: int = 40, rows: int = 20) -> str:
    """ASCII picture of the packing in [0,1]^2 (top row is y near 1).

    Circle i is drawn with glyph i (a-z for circles 0-25), '#' marks cells covered by
    more than one circle, '.' is empty. A circle too small to cover any cell center
    still gets its glyph at its center cell. Parts outside the square are clipped.
    """
    xs = (np.arange(cols) + 0.5) / cols
    ys = 1.0 - (np.arange(rows) + 0.5) / rows
    gx, gy = np.meshgrid(xs, ys)  # (rows, cols)
    d2 = (gx[None] - centers[:, 0, None, None]) ** 2 + (gy[None] - centers[:, 1, None, None]) ** 2
    inside = d2 <= (radii[:, None, None] ** 2)  # (n, rows, cols)
    count = inside.sum(0)
    owner = inside.argmax(0)
    grid = [["." for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if count[r, c] == 1:
                grid[r][c] = _GLYPHS[owner[r, c] % len(_GLYPHS)]
            elif count[r, c] > 1:
                grid[r][c] = "#"
    for i, (x, y) in enumerate(centers):
        if not inside[i].any() and 0 <= x <= 1 and 0 <= y <= 1:
            c, r = min(cols - 1, int(x * cols)), min(rows - 1, int((1 - y) * rows))
            grid[r][c] = _GLYPHS[i % len(_GLYPHS)]
    return "\n".join("".join(row) for row in grid)


def packing_side_info(centers: np.ndarray, radii: np.ndarray, tol: float = TOL) -> dict:
    """Actionable diagnostics for a well-shaped packing (valid or not). JSON-serializable.

    Edge slack of circle i on an edge is the distance it could grow before touching it
    (e.g. x - r for the left edge); pair slack is dist(c_i, c_j) - r_i - r_j. Negative
    slack beyond `tol` is a violation, listed worst first as {"circle", "edge", "by"} or
    {"i", "j", "by"}.
    """
    n = len(radii)
    edges = {
        "left": centers[:, 0] - radii,
        "right": 1.0 - centers[:, 0] - radii,
        "bottom": centers[:, 1] - radii,
        "top": 1.0 - centers[:, 1] - radii,
    }
    edge_viol = [
        {"circle": int(i), "edge": name, "by": _sig(-s[i])}
        for name, s in edges.items() for i in np.nonzero(s < -tol)[0]
    ]
    edge_viol.sort(key=lambda v: -v["by"])
    edge_slack = np.min(np.stack(list(edges.values())), axis=0)  # per circle

    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.sqrt((diff**2).sum(-1))
    slack = dist - radii[:, None] - radii[None, :]
    iu, ju = np.triu_indices(n, k=1)
    pair_slack = slack[iu, ju]
    bad = np.nonzero(pair_slack < -tol)[0]
    overlaps = sorted(
        ({"i": int(iu[k]), "j": int(ju[k]), "by": _sig(-pair_slack[k])} for k in bad),
        key=lambda v: -v["by"],
    )
    min_pair = _sig(pair_slack.min()) if n > 1 else None
    min_edge = _sig(edge_slack.min()) if n else None
    tight_pairs = int((np.abs(pair_slack) <= TIGHT).sum())
    tight_edges = int(sum((np.abs(s) <= TIGHT).sum() for s in edges.values()))
    # Room each circle could grow by alone (min over its edges and neighbours), largest first:
    # a nonzero value means that radius can increase without moving anything.
    own = edge_slack.copy()
    if n > 1:
        s = slack.copy()
        np.fill_diagonal(s, np.inf)
        own = np.minimum(own, s.min(1))
    growable = [
        {"circle": int(i), "room": _sig(own[i])} for i in np.argsort(-own)[:5] if own[i] > TIGHT
    ]
    return {
        "n": n,
        "sum_of_radii": float(radii.sum()),
        "num_edge_violations": len(edge_viol),
        "num_overlaps": len(overlaps),
        "edge_violations": edge_viol[:MAX_LISTED],
        "overlaps": overlaps[:MAX_LISTED],
        "min_edge_slack": min_edge,
        "min_pair_slack": min_pair,
        "tight_contacts": {"edges": tight_edges, "pairs": tight_pairs},
        "growable": growable,
        "radii_range": [_sig(radii.min()), _sig(radii.max())] if n else None,
        "render": render_packing(centers, radii),
    }


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
        expected = {"centers": [N_CIRCLES, 2], "radii": [N_CIRCLES]}
        if isinstance(solution, dict):
            centers, radii = solution.get("centers"), solution.get("radii")
        elif isinstance(solution, list) and len(solution) >= 2:
            centers, radii = solution[0], solution[1]  # extra items (e.g. claimed score) ignored
        else:
            return invalid(
                "construct() must return (centers, radii)",
                side_info={"stage": "shape", "expected": expected, "got_type": type(solution).__name__},
            )
        try:
            c = np.asarray(centers, dtype=float)
            r = np.asarray(radii, dtype=float)
        except (TypeError, ValueError) as e:
            return invalid(
                f"could not convert to arrays: {e}",
                side_info={"stage": "shape", "expected": expected, "message": str(e)[:300]},
            )
        reason = check_packing(c, r)
        well_shaped = (
            c.shape == (N_CIRCLES, 2) and r.shape == (N_CIRCLES,)
            and np.isfinite(c).all() and np.isfinite(r).all()
        )
        if not well_shaped:
            info = {"stage": "shape", "expected": expected,
                    "got": {"centers": list(c.shape), "radii": list(r.shape)}}
            if c.shape == (N_CIRCLES, 2) and r.shape == (N_CIRCLES,):
                info["non_finite"] = {"centers": int((~np.isfinite(c)).any(1).sum()),
                                      "radii": int((~np.isfinite(r)).sum())}
            return invalid(reason or "bad shape", side_info=info)
        side_info = packing_side_info(c, r)
        if (r < 0).any():
            side_info["negative_radii"] = [int(i) for i in np.nonzero(r < 0)[0]]
        if reason:
            side_info["stage"] = "constraint"
            return invalid(reason, side_info=side_info)
        total = float(r.sum())
        return EvalResult(score=total, valid=True, details={"sum_of_radii": total}, side_info=side_info)
