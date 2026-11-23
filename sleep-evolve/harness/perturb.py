"""Small deterministic program perturbations for the eval-matched no_sleep control (extra_evals).

`perturb_program(code, rng, sigma)` multiplies every float literal in `code` by (1 + sigma * N(0, 1))
(rng = random.Random, so a run is reproducible from its seed). Integer literals are left alone:
they are usually counts, sizes or indices, and changing them tends to break the program rather than
explore around it. Only NUMBER tokens are rewritten (via tokenize), so strings and comments are
untouched and the result is still valid Python whenever the input was.

The point is NOT a clever mutation operator: it gives no_sleep the same number of extra evaluator
calls a sleeping arm spends on replays, each one a cheap local probe around the parent, so a
sleep-vs-no-sleep difference cannot be explained by extra sandbox feedback alone.
Returns (new_code, n_changed); n_changed == 0 means the program had no float literal to jitter
(the evaluation is still spent, so the eval count stays matched).
"""

from __future__ import annotations

import io
import math
import random
import tokenize


def _is_float_literal(s: str) -> bool:
    low = s.lower()
    if low.startswith(("0x", "0o", "0b")) or low.endswith("j"):
        return False
    return "." in low or "e" in low


def perturb_program(code: str, rng: random.Random, sigma: float = 0.05) -> tuple[str, int]:
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return code, 0
    edits = []  # (row, col_start, col_end, replacement); NUMBER tokens never span lines
    for tok in toks:
        if tok.type == tokenize.NUMBER and _is_float_literal(tok.string):
            try:
                x = float(tok.string.replace("_", ""))
            except ValueError:
                continue
            # zero gets a small absolute nudge (a relative one would leave it at zero)
            y = x * (1.0 + sigma * rng.gauss(0.0, 1.0)) if x != 0.0 else 0.1 * sigma * rng.gauss(0.0, 1.0)
            if not math.isfinite(y):  # e.g. 1e999 overflow: repr(inf)/repr(nan) is not a valid literal
                continue
            edits.append((tok.start[0], tok.start[1], tok.end[1], repr(float(y))))
    if not edits:
        return code, 0
    lines = code.splitlines(keepends=True)
    for row, c0, c1, rep in reversed(edits):  # right to left keeps earlier columns valid
        ln = lines[row - 1]
        lines[row - 1] = ln[:c0] + rep + ln[c1:]
    return "".join(lines), len(edits)


__all__ = ["perturb_program"]
