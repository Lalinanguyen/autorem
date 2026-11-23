import numpy as np

def _score(pts):
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    n = pts.shape[0]
    if n < 2:
        return -np.inf
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d = dists[mask]
    if d.size == 0 or np.any(np.isinf(d)):
        return -np.inf
    min_dist = float(np.min(d))
    max_dist = float(np.max(d))
    if min_dist <= 0:
        return -np.inf
    ratio2 = (max_dist / min_dist) ** 2
    return -ratio2

def construct():
    rng = np.random.default_rng(42)

    n = 16
    box = 1.0

    def run_once(init_pts, iters=26000, T0=0.35, Tf=0.0008, jitter=0.18):
        pts = init_pts.copy()
        best_pts = pts.copy()
        best_score = _score(pts)

        # Precompute pairwise distances helper
        for k in range(iters):
            t = k / max(1, iters - 1)
            step = T0 * (Tf / max(T0, 1e-9)) ** t
            i = rng.integers(0, n)
            # Larger initial jitter then taper
            delta = rng.normal(scale=step * (1.0 + jitter * rng.random()), size=(2,))
            new_pts = pts.copy()
            new_pts[i] = np.clip(new_pts[i] + delta, 0.0, box)

            new_score = _score(new_pts)
            old_score = _score(pts)

            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts

            # Temperature-based acceptance
            if new_score >= old_score or rng.random() < np.exp((new_score - old_score) / max(step, 1e-9)):
                pts = new_pts

        return best_pts, best_score

    # 1) Strong diverse seeds: 6 runs from well-separated seeds
    best_overall_pts = None
    best_overall_score = -np.inf

    # Seed near a maximally spread 4x4-ish lattice with small jitter
    base_grid = np.array([[i, j] for i in range(4) for j in range(4)], dtype=float) / 3.0

    for s in range(6):
        if s == 0:
            init = base_grid.copy()
            init = init + rng.normal(scale=0.02, size=init.shape)
        elif s == 1:
            init = base_grid[[0,5,10,15,1,6,11,4,9,14,2,7,3,8,12,13]]  # shuffled pattern
            init = init / 1.0
            init = init + rng.normal(scale=0.04, size=init.shape)
        else:
            jitter_seed = rng.random((n, 2)) * 0.25
            init = base_grid + jitter_seed

        init = np.clip(init, 0.0, box)
        pts, score = run_once(init, iters=22000, T0=0.32, Tf=0.0009, jitter=0.22)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_pts = pts

    # 2) Grid seeds with small jitter to refine
    grid = base_grid
    for _ in range(4):
        init = grid + rng.normal(scale=0.05, size=grid.shape)
        init = np.clip(init, 0.0, box)
        pts, score = run_once(init, iters=20000, T0=0.28, Tf=0.0008, jitter=0.25)
        if score > best_overall_score:
            best_overall_score = score
            best_overall_pts = pts

    # 3) Local polishing around best
    if best_overall_pts is not None:
        final_pts, _ = run_once(best_overall_pts, iters=12000, T0=0.18, Tf=0.0005, jitter=0.12)
        best_overall_pts = final_pts

    final = np.asarray(best_overall_pts, dtype=float).reshape(n, 2)
    final[~np.isfinite(final)] = 0.5
    final = np.clip(final, 0.0, box)
    return final
