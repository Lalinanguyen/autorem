"""A cheap, deterministic evaluator for harness tests and local dry runs.

construct() returns a list of DIM floats. score = 1 - mean((x - TARGET)^2), so the
maximum is 1.0 at x == TARGET. It still runs in a subprocess, so the timeout and
crash paths are the real ones.
"""

from __future__ import annotations

import math
from typing import Any

from evaluators.base import EvalResult, SubprocessEvaluator, invalid

DIM = 5
TARGET = 0.7


class MockEvaluator(SubprocessEvaluator):
    problem_id = 0
    name = "mock_quadratic"
    best_known = 1.0
    description = (
        f"Write construct() returning a list of {DIM} floats. "
        f"Score is 1 - mean((x - t)^2) for a hidden target t; higher is better."
    )

    def __init__(self, timeout_s: int = 10):
        self.timeout_s = timeout_s

    def seed_program(self) -> str:
        return f"def construct():\n    return [0.0] * {DIM}\n"

    def score_solution(self, solution: Any) -> EvalResult:
        if not isinstance(solution, list) or len(solution) != DIM:
            got = {"got_type": type(solution).__name__}
            if isinstance(solution, (list, dict)):
                got["got_len"] = len(solution)
            return invalid(
                f"expected a list of {DIM} numbers",
                side_info={"stage": "shape", "expected": f"list of {DIM} floats", **got},
            )
        try:
            xs = [float(v) for v in solution]
        except (TypeError, ValueError):
            bad = [i for i, v in enumerate(solution) if not isinstance(v, (int, float))]
            return invalid("non-numeric value", side_info={"stage": "shape", "non_numeric_indices": bad})
        if not all(math.isfinite(v) for v in xs):
            bad = [i for i, v in enumerate(xs) if not math.isfinite(v)]
            return invalid("non-finite value", side_info={"stage": "constraint", "non_finite_indices": bad})
        sq = [(v - TARGET) ** 2 for v in xs]
        score = 1.0 - sum(sq) / DIM
        worst = max(range(DIM), key=lambda i: sq[i])
        side_info = {
            "per_coordinate_sq_error": [round(e, 6) for e in sq],
            "worst_coordinate": worst,
            "mean_sq_error": round(sum(sq) / DIM, 6),
        }
        return EvalResult(score=score, valid=True, side_info=side_info)
