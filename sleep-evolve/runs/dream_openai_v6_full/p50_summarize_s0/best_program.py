import numpy as np

def _ratio_sq(points):
    n = points.shape[0]
    diff = points[:, None, :] - points[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    triu = d2[np.triu_indices(n, k=1)]
    d2_min = np.min(triu)
    d2_max = np.max(triu)
    if d2_min <= 0:
        return np.inf
    return d2_max / d2_min

def _normalize_min1(points):
    # Normalize so that the minimum pairwise distance is 1
    n = points.shape[0]
    diff = points[:, None, :] - points[None, :, :]
    d2 = (diff * diff).sum(axis=2)
    triu = d2[np.triu_indices(n, k=1)]
    d2_min = np.min(triu)
    if d2_min <= 0 or not np.isfinite(d2_min):
        # tiny jitter to escape degeneracy
        pts = points.copy()
        pts += 1e-6
        return pts
    dist_min = np.sqrt(d2_min)
    return points / dist_min

def _init_seed_grid(n=16, scale=1.0):
    pts = np.array([[i, j] for i in range(4) for j in range(4)], dtype=float)
    pts *= scale
    return pts

def _init_seed_two_rings(n=16, inner=8, outer=8, inner_r=0.9, outer_r=2.0):
    pts = []
    for k in range(inner):
        ang = 2 * np.pi * k / inner
        pts.append([inner_r * np.cos(ang), inner_r * np.sin(ang)])
    for k in range(outer):
        ang = 2 * np.pi * k / outer
        pts.append([outer_r * np.cos(ang), outer_r * np.sin(ang)])
    pts = np.array(pts, dtype=float)
    if pts.shape[0] > n:
        pts = pts[:n]
    elif pts.shape[0] < n:
        extra = _init_seed_grid(n - pts.shape[0], scale=0.5)
        pts = np.vstack([pts, extra])
    return pts

def _perturb(points, rng, step):
    pts = points.copy()
    i = rng.integers(0, pts.shape[0])
    delta = rng.normal(scale=step, size=2)
    pts[i] += delta
    return pts

def _random_smart_jitter(points, rng, scale=0.03):
    pts = points.copy()
    pts += rng.normal(scale=scale, size=pts.shape)
    return pts

def construct():
    rng = np.random.default_rng(987654321)

    n = 16

    # Build a diverse set of seeds around a known good basin (p7-ish)
    seeds = []
    seeds.append(_init_seed_grid(n, scale=1.2))
    seeds.append(_init_seed_grid(n, scale=0.9))
    seeds.append(_init_seed_two_rings(n, inner=8, outer=8, inner_r=0.9, outer_r=2.0))

    # Create multiple small perturbed copies around the seeds (focus on p7-like basin)
    perturbed = []
    for base in seeds:
        for _ in range(6):  # 6 perturbations per seed
            s = base.copy()
            s = _normalize_min1(s)
            # perturb with small sigma tuned for local exploration
            s = _random_smart_jitter(s, rng, scale=0.03)
            s = _normalize_min1(s)
            perturbed.append(s)
    seeds = seeds + perturbed

    # Normalize all seeds to min distance 1
    seeds = [_normalize_min1(s) for s in seeds]

    # If any seed isn't exactly (16,2), trim/pad deterministically
    for i in range(len(seeds)):
        s = seeds[i]
        if s.shape[0] != n or s.shape[1] != 2:
            s = np.asarray(s, dtype=float).reshape(-1, 2)
            if s.shape[0] != n:
                if s.shape[0] > n:
                    s = s[:n]
                else:
                    pad = np.zeros((n - s.shape[0], 2), dtype=float)
                    s = np.vstack([s, pad])
            seeds[i] = s

    # SA parameters
    iterations_per_seed = 9000
    initial_step = 0.035  # small moves around p7 basin
    min_step = 1e-4
    reduce_every = 1500
    temp0 = 0.12
    temp_decay = 0.9996

    best_all = None
    best_ratio_all = np.inf

    for s_idx, seed in enumerate(seeds):
        pts = seed.copy()
        if pts.shape != (n, 2):
            pts = _normalize_min1(pts)
            if pts.shape != (n, 2):
                pts = _init_seed_grid(n, scale=1.0)

        best = pts.copy()
        best_ratio = _ratio_sq(best)

        step = max(initial_step * (0.98 ** s_idx), min_step)
        temp = temp0

        # Early intensification: keep a small elite across iterations
        for it in range(iterations_per_seed):
            new_pts = best.copy()
            # With small probability, perform a small block move (2-4 points)
            if (it % 1000) == 0:
                k = 2
                idx = np.arange(n)
                rng.shuffle(idx)
                ids = idx[:k]
                delta = rng.normal(scale=step*2, size=(k, 2))
                new_pts[ids] += delta

            # Regular single-point perturbation
            new_pts = _perturb(new_pts, rng, step)
            new_pts = _normalize_min1(new_pts)

            r = _ratio_sq(new_pts)
            if r < best_ratio or rng.random() < np.exp((best_ratio - r) / max(1e-9, temp)):
                best = new_pts
                best_ratio = r

            # occasional larger jitter to escape flats
            if it % 2000 == 0 and it > 0:
                trial = best + rng.normal(scale=0.05, size=(n, 2))
                trial = _normalize_min1(trial)
                r2 = _ratio_sq(trial)
                if r2 < best_ratio:
                    best = trial
                    best_ratio = r2

            if (it + 1) % reduce_every == 0 and step > min_step:
                step *= 0.92

            temp *= temp_decay

            # Early stop if already very good
            if best_ratio <= 12.9:
                break

        if best_ratio < best_ratio_all:
            best_all = best.copy()
            best_ratio_all = best_ratio

    if best_all is None:
        best_all = _init_seed_grid(n, scale=1.0)

    best_all = _normalize_min1(best_all)

    best_all = np.asarray(best_all, dtype=float)
    if best_all.shape != (n, 2):
        best_all = best_all[:n, :2]

    # Final safety: finite
    best_all = np.where(np.isfinite(best_all), best_all, 0.0)

    return best_all
