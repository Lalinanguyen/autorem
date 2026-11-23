import numpy as np

def _ratio_from_points(pts):
    pts = np.asarray(pts)
    n = pts.shape[0]
    diff = pts[:, None, :] - pts[None, :, :]
    dist_matrix = np.sqrt((diff ** 2).sum(-1))
    iu1, iu2 = np.triu_indices(n, k=1)
    d = dist_matrix[iu1, iu2]
    dmin = d.min()
    dmax = d.max()
    if dmin <= 0.0 or not np.isfinite(dmin) or not np.isfinite(dmax):
        return np.inf
    return dmax / dmin

def _score(pts):
    ratio = _ratio_from_points(pts)
    if not np.isfinite(ratio) or ratio <= 0:
        return np.inf
    return ratio ** 2

def construct():
    np.random.seed(12345)
    n = 16

    # Start from a well-spread 4x4 grid in [0,1] with modest jitter
    grid_size = 4
    xs = np.linspace(0.15, 0.85, grid_size)
    ys = np.linspace(0.15, 0.85, grid_size)
    base_pts = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    pts = base_pts + (np.random.rand(n, 2) - 0.5) * 0.03  # small jitter
    pts = np.clip(pts, 0.0, 1.0)

    curr_score = -_score(pts)  # higher is better
    best_pts = pts.copy()
    best_score = curr_score

    max_iters = 30000
    temp = 0.25
    cool_rate = 0.9997
    stagnation_limit = 4000
    no_improve = 0

    for it in range(max_iters):
        temp = max(1e-4, temp * cool_rate)

        # Single-point move: propose a small random displacement
        idx = np.random.randint(0, n)
        old_pt = pts[idx].copy()
        delta = (np.random.rand(2) * 2 - 1) * temp
        pts[idx] = old_pt + delta
        pts = np.clip(pts, 0.0, 1.0)

        new_score = -_score(pts)

        if new_score >= curr_score:
            curr_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = pts.copy()
                no_improve = 0
        else:
            # revert
            pts[idx] = old_pt
            no_improve += 1

        if best_score >= -12.89:
            break
        if no_improve >= stagnation_limit:
            # Restart from a fresh grid-like state to escape plateaus
            pts = base_pts + (np.random.rand(n, 2) - 0.5) * 0.04
            pts = np.clip(pts, 0.0, 1.0)
            curr_score = -_score(pts)
            best_pts = pts.copy()
            best_score = curr_score
            no_improve = 0
            temp = 0.25

    final_pts = np.asarray(best_pts, dtype=float).reshape(n, 2)
    final_pts = np.clip(final_pts, 0.0, 1.0)
    return final_pts
