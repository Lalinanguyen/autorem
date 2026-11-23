"""Problem 7: difference bases for integer intervals, minimizing |B|^2 / n.

Source: google-deepmind/alphaevolve_repository_of_problems,
experiments/difference_bases/difference_bases.ipynb (problem page problems/7.html;
paper arXiv:2511.02864, Problem 6.7).

The notebook's get_score: coerce every element with int(), dedupe and sort, append 0 if
missing, reject sets larger than 2000, then take the positive differences B[j] - B[i].
n is the largest integer with 1..n all differences of B; the quantity is |B|^2 / n
(minimize). Elements are non-negative integers per the instance params. The notebook's
-100 sentinel for "cannot cover 1" maps to an invalid result here.

Registry direction is minimize, so the higher-is-better score is -(|B|^2 / n).
best_known is AlphaEvolve's Leech combination A=[0,1,4,6] x B (90-element Singer set)
with m = 89^2+89+1: |L| = 360 covering 1..49109 -> 2.639027469506608.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from evaluators.base import EvalResult, SubprocessEvaluator, invalid

MAX_SET_SIZE = 2000
BEST_KNOWN = -2.639027469506608  # score is the negated quantity |B|^2 / n

_SEED = '''def construct():
    """Baseline: B = {0, 1, ..., 20}, covering 1..20 with |B|^2/n = 22.05."""
    return list(range(21))
'''


def coverage(b_sorted: np.ndarray) -> tuple[int, int]:
    """(n, first_uncovered): largest n with 1..n all differences of B, and the first miss.

    Differences are b - a for a < b in B (sorted, so positive). first_uncovered is n + 1,
    or 1 when even the difference 1 is missing.
    """
    if len(b_sorted) < 2:
        return 0, 1
    diffs = np.unique((b_sorted[None, :] - b_sorted[:, None])[np.triu_indices(len(b_sorted), k=1)])
    if len(diffs) == 0 or diffs[0] != 1:
        return 0, 1
    gaps = np.nonzero(np.diff(diffs) > 1)[0]
    if len(gaps):
        n = int(diffs[gaps[0]])
        return n, n + 1
    n = int(diffs[-1])
    return n, n + 1


class DifferenceBasis(SubprocessEvaluator):
    problem_id = 7
    name = "p7_difference_basis"
    best_known = BEST_KNOWN
    benchmark_instance = {"set": "nonnegative_integers", "n": "free", "max_set_size": 2000}
    description = (
        "Find a small set B of non-negative integers whose differences B - B cover every "
        "integer in [1, n] for the largest possible n, minimizing |B|^2 / n. Write a Python "
        "program defining construct() that returns a list of non-negative integers (at "
        f"most {MAX_SET_SIZE} elements after deduplication; 0 is added if missing). "
        "Score = -(|B|^2 / n) where n is the largest integer such that every d in 1..n "
        f"is a difference of two elements of B (best known: {-BEST_KNOWN:.9f}, i.e. score "
        f"{BEST_KNOWN:.9f}). A set that cannot cover the difference 1 is invalid. numpy and "
        "the standard library are available. construct() must finish within the time limit."
    )

    def __init__(self, timeout_s: int = 60):
        self.timeout_s = timeout_s

    def seed_program(self) -> str:
        return _SEED

    def score_solution(self, solution: Any) -> EvalResult:
        raw = solution.get("basis") if isinstance(solution, dict) else solution
        if not isinstance(raw, (list, tuple)):
            return invalid(
                "construct() must return a list of non-negative integers",
                side_info={"stage": "shape", "expected": "list of ints",
                           "got_type": type(solution).__name__},
            )
        bad = [i for i, x in enumerate(raw) if isinstance(x, bool) or not isinstance(x, (int, float))
               or (isinstance(x, float) and not math.isfinite(x))]
        if bad:
            return invalid(
                "non-numeric element in the basis",
                side_info={"stage": "shape", "bad_indices": bad[:10]},
            )
        b = sorted({int(x) for x in raw})
        if not b:
            return invalid("the basis is empty", side_info={"stage": "shape"})
        neg = [int(i) for i, x in enumerate(b) if x < 0]
        if neg:
            return invalid(
                "elements must be non-negative integers",
                side_info={"stage": "constraint", "negative_elements": [b[i] for i in neg[:10]]},
            )
        if 0 not in b:
            b.append(0)
        if len(b) > MAX_SET_SIZE:
            return invalid(
                f"|B| = {len(b)} exceeds the maximum {MAX_SET_SIZE}",
                side_info={"stage": "constraint", "size": len(b), "max_set_size": MAX_SET_SIZE},
            )
        b_arr = np.asarray(sorted(b), dtype=np.int64)
        n, first_uncovered = coverage(b_arr)
        side_info = {
            "size": len(b),
            "coverage_n": n,
            "first_uncovered": first_uncovered,
            "min_element": int(b_arr[0]),
            "max_element": int(b_arr[-1]),
            "quantity": len(b) ** 2 / n if n else None,
        }
        if n == 0:
            return invalid(
                "the basis does not cover the difference 1 (n = 0)",
                side_info={"stage": "constraint", **side_info},
            )
        score = -(len(b) ** 2 / n)
        return EvalResult(score=score, valid=True,
                          details={"size": len(b), "coverage_n": n, "quantity": -score},
                          side_info=side_info)
