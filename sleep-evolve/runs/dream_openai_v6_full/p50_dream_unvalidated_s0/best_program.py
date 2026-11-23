import numpy as np

def _ratio_squared(points):
    pts = np.asarray(points, dtype=float)
    n = pts.shape[0]
    if n <= 1:
        return np.inf
    d = pts[:, None, :] - pts[None, :, :]
    dist2 = np.einsum('ijk,ijk->ij', d, d)
    mask = ~np.eye(n, dtype=bool)
    d2 = dist2[mask]
    if d2.size == 0:
        return np.inf
    min_d2 = d2.min()
    max_d2 = dist2.max()
    if min_d2 <= 0:
        return np.inf
    return max_d2 / min_d2

def _grid4_unit_square(n=16):
    # 4x4 grid on [0,1]x[0,1]
    pts = np.array([[x, y] for x in np.linspace(0.0, 1.0, 4) for y in np.linspace(0.0, 1.0, 4)])
    return pts[:n]

def _clamp(a, lo=0.0, hi=1.0):
    return np.clip(a, lo, hi)

def _min_pair_distance(points):
    pts = np.asarray(points, dtype=float)
    n = pts.shape[0]
    if n < 2:
        return np.inf
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(d, axis=-1)
    mask = ~np.eye(n, dtype=bool)
    dmin = dist[mask].min()
    return dmin

def construct():
    rng = np.random.default_rng(seed=12345)
    n = 16
    target_min_sep = 0.22  # guard: require at least this separation to proceed
    max_trials = 1800
    steps = [0.18, 0.12, 0.08, 0.04, 0.02]

    # Initial seed: jittered 4x4 grid
    base = _grid4_unit_square(n=n)
    init = base.copy()
    init = init + 0.035 * rng.standard_normal(init.shape)
    init = _clamp(init, 0.0, 1.0)

    # Ensure finite
    if not np.all(np.isfinite(init)):
        init = rng.random((n, 2))

    def feasible(cur):
        cur = np.asarray(cur, dtype=float)
        cur = _clamp(cur, 0.0, 1.0)
        # small jitter to avoid pathological exact collisions
        cur += 1e-9 * rng.standard_normal(size=cur.shape)
        return cur

    def guard_minsep(cur, min_sep):
        cur = np.asarray(cur, dtype=float)
        dmin = _min_pair_distance(cur)
        return dmin >= min_sep, dmin

    best = init.copy()
    # Enforce initial guard
    ok, dmin0 = guard_minsep(best, target_min_sep)
    if not ok or not np.isfinite(_ratio_squared(best)):
        best = _grid4_unit_square(n=n) + 0.01 * rng.standard_normal((n, 2))
        best = _clamp(best, 0.0, 1.0)

    best_score = _ratio_squared(best)

    # If initial best violates min-sep, repair with small jitter until ok
    if not ok:
        for _ in range(200):
            best = best + 0.02 * rng.standard_normal((n, 2))
            best = feasible(best)
            ok, dmin0 = guard_minsep(best, target_min_sep)
            if ok:
                best_score = _ratio_squared(best)
                break

    # Multi-start hill-climb with hard min-sep constraint
    for r in range(6):
        if r == 0:
            cur = best + 0.05 * rng.standard_normal((n, 2))
        elif r < 3:
            cur = init + 0.08 * rng.standard_normal((n, 2))
        else:
            cur = rng.random((n, 2))
        cur = feasible(cur)
        ok, dmin_cur = guard_minsep(cur, target_min_sep)
        if not ok or not np.isfinite(_ratio_squared(cur)):
            cur = rng.random((n, 2))
            cur = feasible(cur)
            ok, dmin_cur = guard_minsep(cur, target_min_sep)
        if not ok:
            cur = init.copy()
            cur = feasible(cur)
            ok, dmin_cur = guard_minsep(cur, target_min_sep)

        cur_score = _ratio_squared(cur)
        if not np.isfinite(cur_score) or cur_score <= 0:
            cur = init.copy()
            cur = feasible(cur)
            cur_score = _ratio_squared(cur)

        for s in steps:
            for _ in range(max_trials):
                i = rng.integers(0, n)
                cand = cur.copy()
                scale = s
                cand[i, :] += rng.normal(scale=scale, size=2)
                cand = feasible(cand)
                ok, dmin_cand = guard_minsep(cand, target_min_sep)
                if not ok:
                    continue
                c_score = _ratio_squared(cand)
                if not np.isfinite(c_score) or c_score <= 0:
                    continue

                # Accept if improvement, else probabilistically escape
                if c_score < cur_score - 1e-12:
                    cur = cand
                    cur_score = c_score
                    if cur_score < best_score:
                        best = cur.copy()
                        best_score = cur_score
                else:
                    delta = c_score - cur_score
                    T = max(1e-6, 0.04 * (1.0 - (s / max(steps))))
                    if rng.random() < np.exp(-delta / max(T, 1e-9)):
                        cur = cand
                        cur_score = c_score

        # Diversify between restarts
        if r % 2 == 0:
            cur = best + 0.07 * rng.standard_normal((n, 2))
        else:
            cur = rng.random((n, 2))
        cur = feasible(cur)
        ok, dmin_cur = guard_minsep(cur, target_min_sep)
        if not ok:
            cur = init + 0.05 * rng.standard_normal((n, 2))
            cur = feasible(cur)
            ok, dmin_cur = guard_minsep(cur, target_min_sep)
        cur_score = _ratio_squared(cur)
        if not np.isfinite(cur_score) or cur_score <= 0:
            cur = rng.random((n, 2))
            cur = feasible(cur)
            cur_score = _ratio_squared(cur)

        if best is not None:
            d = best[:, None, :] - best[None, :, :]
            dist = np.linalg.norm(d, axis=-1)
            min_d = dist[~np.eye(n, dtype=bool)].min() if n > 1 else 0.0
            if not np.isfinite(best_score) or min_d <= 1e-4:
                best = _grid4_unit_square(n=n) + 0.01 * rng.standard_normal((n, 2))
                best = _clamp(best, 0.0, 1.0)
                best_score = _ratio_squared(best)

        # Ensure a reasonable finish
        if not np.all(np.isfinite(best)):
            best = rng.random((n, 2))

    # Final sanity: enforce min-sep on best; if still degenerate, jitter slightly
    ok, dmin_final = guard_minsep(best, target_min_sep)
    if not ok or not np.isfinite(best_score) or best_score <= 0:
        best = _grid4_unit_square(n=n) + 0.02 * rng.standard_normal((n, 2))
        best = _clamp(best, 0.0, 1.0)
        ok, dmin_final = guard_minsep(best, target_min_sep)
        if not ok:
            # as a last resort, random feasible with min_sep
            for _ in range(500):
                candidate = rng.random((n, 2))
                candidate = feasible(candidate)
                if guard_minsep(candidate, target_min_sep)[0]:
                    best = candidate
                    best_score = _ratio_squared(best)
                    break

    # Ensure finiteness and return
    if not np.all(np.isfinite(best)):
        best = rng.random((n, 2))

    return best
