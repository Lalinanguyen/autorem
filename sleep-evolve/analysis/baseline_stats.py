"""Paired statistics in numpy only (scipy is not a dependency).

wilcoxon_signed_rank(d)
  Two-sided Wilcoxon signed-rank test of H0: the paired differences are symmetric around 0.
  - Zero differences (|d| <= zero_tol) are dropped (Wilcoxon's original method, scipy's default
    zero_method="wilcox"); their count is reported as n_zero.
  - |d| are ranked with average ranks for ties. W+ = sum of the ranks of positive differences.
  - n <= exact_max_n (default 20) non-zero pairs: EXACT null distribution of W+ over all 2^n
    equally likely sign assignments, using the actual (tie-averaged) ranks. It is computed by
    convolving one rank at a time over doubled (integer) ranks, which enumerates the 2^n
    assignments' sums exactly. p = P(|W+ - n(n+1)/4| >= |w - n(n+1)/4|).
  - Larger n: normal approximation with the tie correction of the variance,
    var = n(n+1)(2n+1)/24 - sum(t^3 - t)/48 over tie groups, and a 0.5 continuity correction.
  With n non-zero pairs the smallest attainable exact two-sided p is 2 / 2^n (n=5: 0.0625), so
  fewer than 6 informative pairs can never be significant at 0.05.

bootstrap_median_ci(d)
  Percentile bootstrap CI of the median paired difference (seeded numpy Generator, resampling
  pairs with replacement).

win_tie_loss(d, better)
  Counts across pairs from the first arm's point of view; `better` is "higher" or "lower".
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _clean(diffs: Iterable[float | None]) -> np.ndarray:
    d = np.asarray([x for x in diffs if x is not None], dtype=float)
    return d[np.isfinite(d)]


def average_ranks(x: np.ndarray) -> np.ndarray:
    """1-based ranks of x with ties given their average rank."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def wilcoxon_signed_rank(diffs: Iterable[float | None], *, exact_max_n: int = 20, zero_tol: float = 1e-12) -> dict:
    d = _clean(diffs)
    nz = d[np.abs(d) > zero_tol]
    n, n_zero = len(nz), len(d) - len(nz)
    if n == 0:
        return {"p": 1.0, "w_plus": 0.0, "n": 0, "n_zero": n_zero, "method": "no non-zero differences", "z": None}
    absd = np.abs(nz)
    ranks = average_ranks(absd)
    w_plus = float(ranks[nz > 0].sum())
    mean = n * (n + 1) / 4.0
    if n <= exact_max_n:
        r2 = np.rint(2 * ranks).astype(int)  # tie-averaged ranks are multiples of 0.5
        total = int(r2.sum())
        dist = np.zeros(total + 1, dtype=float)
        dist[0] = 1.0
        for r in r2:  # each rank is either in W+ or not: new[k] = old[k] + old[k - r]
            shifted = np.zeros_like(dist)
            shifted[r:] = dist[:-r] if r else dist
            dist = dist + shifted
        probs = dist / dist.sum()
        k = np.arange(total + 1) / 2.0
        p = float(probs[np.abs(k - mean) >= abs(w_plus - mean) - 1e-9].sum())
        return {"p": min(1.0, p), "w_plus": w_plus, "n": n, "n_zero": n_zero, "z": None,
                "method": f"exact (2^{n} sign assignments, tie-averaged ranks, zeros dropped)"}
    _, counts = np.unique(absd, return_counts=True)
    var = n * (n + 1) * (2 * n + 1) / 24.0 - float(((counts ** 3) - counts).sum()) / 48.0
    if var <= 0:
        return {"p": 1.0, "w_plus": w_plus, "n": n, "n_zero": n_zero, "z": None, "method": "normal (zero variance)"}
    dev = abs(w_plus - mean) - 0.5
    z = max(0.0, dev) / math.sqrt(var)
    p = math.erfc(z / math.sqrt(2.0))
    return {"p": min(1.0, p), "w_plus": w_plus, "n": n, "n_zero": n_zero, "z": z,
            "method": "normal approximation (tie-corrected variance, continuity 0.5, zeros dropped)"}


def bootstrap_median_ci(diffs: Iterable[float | None], *, n_boot: int = 10000, seed: int = 0,
                        level: float = 0.95) -> tuple[float | None, float | None, float | None]:
    """(median, lo, hi). None for no data; a single pair gives a degenerate interval."""
    d = _clean(diffs)
    if len(d) == 0:
        return None, None, None
    med = float(np.median(d))
    if len(d) == 1:
        return med, med, med
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    meds = np.median(d[idx], axis=1)
    a = (1.0 - level) / 2.0
    return med, float(np.quantile(meds, a)), float(np.quantile(meds, 1.0 - a))


def win_tie_loss(diffs: Iterable[float | None], better: str = "higher", tol: float = 1e-9) -> dict:
    if better not in ("higher", "lower"):
        raise ValueError("better must be 'higher' or 'lower'")
    d = _clean(diffs)
    sign = 1.0 if better == "higher" else -1.0
    s = sign * d
    return {"win": int((s > tol).sum()), "tie": int((np.abs(s) <= tol).sum()), "loss": int((s < -tol).sum())}


__all__ = ["average_ranks", "bootstrap_median_ci", "wilcoxon_signed_rank", "win_tie_loss"]
