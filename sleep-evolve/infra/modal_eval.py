"""Evaluators on Modal CPU containers.

Deploy once:   uv run modal deploy infra/modal_eval.py
Then set `eval_backend: modal` in a run config to evaluate remotely from a local run.
(Runs launched by scripts/run_matrix.py already execute inside a Modal CPU container,
so they evaluate locally there, still inside a subprocess with a timeout.)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import modal  # noqa: E402

from evaluators.base import EvalResult, Evaluator, invalid  # noqa: E402

APP_NAME = "sleep-evolve-eval"
FUNCTION_NAME = "evaluate_remote"
MAX_EVAL_TIMEOUT_S = 900  # container ceiling; per-problem timeout_s is enforced inside

app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.26", "pydantic>=2.6")
    .add_local_python_source("evaluators")
)


@app.function(image=image, cpu=1.0, memory=2048, timeout=MAX_EVAL_TIMEOUT_S)
def evaluate_remote(problem_id: int, program: str, timeout_s: int | None = None) -> dict:
    from evaluators import make_evaluator

    kwargs = {"timeout_s": timeout_s} if timeout_s else {}
    return make_evaluator(problem_id, **kwargs).evaluate(program).model_dump()


class ModalEvaluator:
    """Evaluator that forwards evaluate() to the deployed Modal function. Same metadata as `local`."""

    def __init__(self, local: Evaluator):
        self.problem_id = local.problem_id
        self.name = local.name
        self.best_known = local.best_known
        self.timeout_s = local.timeout_s
        self.description = local.description
        self._local = local
        self._fn = modal.Function.from_name(APP_NAME, FUNCTION_NAME)

    def seed_program(self) -> str:
        return self._local.seed_program()

    def evaluate(self, program: str) -> EvalResult:
        try:
            return EvalResult(**self._fn.remote(self.problem_id, program, self.timeout_s))
        except Exception as e:  # network / Modal errors must not crash the loop
            return invalid(f"modal evaluation failed: {type(e).__name__}: {e}")
