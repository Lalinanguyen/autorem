import numpy as np


def construct():
    """Baseline: 26 equal circles on a 6x5 grid. Returns (centers, radii)."""
    cols, rows, n = 6, 5, 26
    r = 0.5 / max(cols, rows)
    centers = []
    for k in range(n):
        i, j = divmod(k, cols)
        centers.append([(j + 0.5) / cols, (i + 0.5) / rows])
    return np.array(centers), np.full(n, r)
