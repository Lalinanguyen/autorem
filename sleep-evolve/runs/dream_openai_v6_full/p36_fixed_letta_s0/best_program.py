import numpy as np

def compute_radii(centers, tol=1e-9, max_iters=8000):
    """
    Given fixed centers, compute feasible radii by border and pairwise constraints.
    Uses an iterative projection-like relaxation that only grows radii when feasible.
    """
    centers = np.asarray(centers, dtype=float)
    n = centers.shape[0]
    radii = np.zeros(n, dtype=float)

    # Precompute border caps for each circle
    left = centers[:, 0]
    right = 1.0 - centers[:, 0]
    bottom = centers[:, 1]
    top = 1.0 - centers[:, 1]
    border_cap = np.minimum(np.minimum(left, right), np.minimum(bottom, top))

    # Precompute pairwise distances
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            Dij = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            D[i, j] = D[j, i] = Dij

    improved = True
    it = 0
    while improved and it < max_iters:
        improved = False
        it += 1
        for i in range(n):
            # Maximum allowed by border
            max_by_border = border_cap[i]
            # Maximum allowed by others
            max_by_others = max_by_border
            for j in range(n):
                if i == j:
                    continue
                # Enforce non-overlap: r_i <= dist(i,j) - r_j
                max_by_others = min(max_by_others, D[i, j] - radii[j])
            new_r = max(0.0, min(max_by_border, max_by_others))
            if new_r > radii[i] + tol:
                radii[i] = new_r
                improved = True

    # Final adjustment to ensure strict feasibility
    for i in range(n):
        for j in range(i + 1, n):
            d = D[i, j]
            if radii[i] + radii[j] > d + 1e-9:
                excess = radii[i] + radii[j] - d
                if radii[i] >= radii[j]:
                    radii[i] = max(0.0, radii[i] - excess)
                else:
                    radii[j] = max(0.0, radii[j] - excess)

    radii = np.minimum(radii, border_cap)
    return radii

def construct():
    rng = np.random.default_rng(12345)

    # Deterministic hex-like seed (26 centers)
    centers = []
    row_counts = [6, 5, 6, 5, 4]  # sums to 26
    nrows = len(row_counts)

    # Layout bounds
    y0, y1 = 0.12, 0.88
    y_positions = np.linspace(y0, y1, nrows)

    for r, cy in enumerate(y_positions):
        m = row_counts[r]
        # Horizontal span per row
        xs = np.linspace(0.15, 0.85, m)
        # Stagger every other row
        offset = (r % 2) * 0.08
        xs = xs + offset
        for x in xs:
            # small deterministic jitter to avoid perfect symmetry
            xj = x + rng.normal(scale=0.0015)
            yj = cy + rng.normal(scale=0.0015)
            centers.append((float(np.clip(xj, 0.0, 1.0)),
                            float(np.clip(yj, 0.0, 1.0))))
    centers = np.array(centers, dtype=float)
    assert centers.shape == (26, 2)

    # Compute radii from this seed
    radii = compute_radii(centers, max_iters=8000)
    # Respect border constraints again as a final guard
    radii = np.minimum(radii, np.minimum(centers[:, 0], 1.0 - centers[:, 0]))
    radii = np.minimum(radii, np.minimum(centers[:, 1], 1.0 - centers[:, 1]))

    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = float(np.sum(radii))

    # Focused local search: move a single center a bit to potentially improve total radii
    moves = 1800
    max_move = 0.08
    for t in range(moves):
        i = rng.integers(0, 26)
        step = max_move * (0.98 ** (t // 120))
        dx = rng.uniform(-step, step)
        dy = rng.uniform(-step, step)
        old_center = centers[i].copy()
        centers[i, 0] = float(np.clip(centers[i, 0] + dx, 0.0, 1.0))
        centers[i, 1] = float(np.clip(centers[i, 1] + dy, 0.0, 1.0))

        new_radii = compute_radii(centers, max_iters=8000)
        new_radii = np.minimum(new_radii, np.minimum(centers[:, 0], 1.0 - centers[:, 0]))
        new_radii = np.minimum(new_radii, np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
        s = float(np.sum(new_radii))

        if s > best_sum + 1e-12:
            best_sum = s
            best_centers = centers.copy()
            best_radii = new_radii.copy()
        else:
            centers[i] = old_center  # revert if not improving

    return best_centers, best_radii
