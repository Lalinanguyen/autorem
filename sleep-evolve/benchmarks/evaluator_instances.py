"""The instance each of OUR evaluators actually scores, for comparison with the registry.

An evaluator may declare its instance itself with a `benchmark_instance: dict` attribute (preferred
for new evaluators). For evaluators that predate this layer (evaluators/ is not ours to edit), the
declaration lives here and is derived from the evaluator module's own constants where they exist.

The registry instance is chosen by exact equality of params; a mismatch means "refuse to compare".
"""

from __future__ import annotations

from typing import Any, Callable

from evaluators.base import Evaluator


def _p36(ev: Evaluator) -> dict[str, Any]:
    from evaluators import p36_circle_packing as m

    # check_packing tests containment in [0,1]^2 (edges at 0 and 1): the unit square.
    return {"n": int(m.N_CIRCLES), "container": "unit_square"}


DECLARED: dict[int, Callable[[Evaluator], dict[str, Any]]] = {
    36: _p36,
}


def declared_instance(evaluator: Evaluator) -> dict[str, Any] | None:
    """The evaluator's instance params, or None if it declares none (scoring is then refused)."""
    own = getattr(evaluator, "benchmark_instance", None)
    if isinstance(own, dict) and own:
        return dict(own)
    fn = DECLARED.get(int(getattr(evaluator, "problem_id", -1)))
    return fn(evaluator) if fn else None
