import numpy as np

def compute_radii_ordered(centers):
    """
    centers: (n,2) array
    returns radii (n,) or None if infeasible
    Radii are the maximum non-overlapping radii given the fixed center order.
    """
    n = centers.shape[0]
    radii = np.empty(n, dtype=float)
    for i in range(n):
        x, y = centers[i]
        r = min(x, y, 1.0 - x, 1.0 - y)  # wall constraint
        for j in range(i):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.hypot(dx, dy)
            r = min(r, dist - radii[j])
        if r < 0:
            return None
        radii[i] = r
    return radii

def clamp_centers(centers, eps=1e-8):
    return np.clip(centers, eps, 1.0 - eps)

def random_center(n):
    return np.random.uniform(0.0, 1.0, size=(n, 2))

def is_feasible(centers):
    return compute_radii_ordered(centers) is not None

def construct():
    np.random.seed(20241123)
    n = 26
    best_centers = None
    best_radii = None
    best_sum = -np.inf

    # Helper: single randomized attempt with local search
    def attempt(init_centers, iter_limit=6000):
        nonlocal best_centers, best_radii, best_sum
        centers = init_centers.copy()
        radii = compute_radii_ordered(centers)
        if radii is None:
            return None, None, -np.inf
        cur_sum = float(np.sum(radii))

        # annealing-like moves
        for it in range(iter_limit):
            i = np.random.randint(0, n)
            old = centers[i].copy()
            # adaptive step size
            t = max(0.02, 1.0 - it / iter_limit)
            step = 0.10 * t
            centers[i] += np.random.normal(scale=step, size=2)
            centers[i] = np.clip(centers[i], 0.0 + 1e-6, 1.0 - 1e-6)

            r_new = compute_radii_ordered(centers)
            if r_new is not None:
                s_new = float(np.sum(r_new))
                # accept if improves or with small chance to escape
                if s_new >= cur_sum or np.random.rand() < 0.01:
                    radii = r_new
                    cur_sum = s_new
                    if s_new > best_sum:
                        best_sum = s_new
                        best_centers = centers.copy()
                        best_radii = r_new.copy()
                else:
                    centers[i] = old
            else:
                centers[i] = old

            # small perturb around current best to explore plateau
            if it % 1000 == 0 and best_centers is not None:
                for _ in range(3):
                    j = np.random.randint(0, n)
                    oldb = best_centers[j].copy()
                    best_centers[j] += np.random.normal(scale=0.03, size=2)
                    best_centers[j] = np.clip(best_centers[j], 0.0 + 1e-6, 1.0 - 1e-6)
                    rb = compute_radii_ordered(best_centers)
                    if rb is not None:
                        srb = float(np.sum(rb))
                        if srb > best_sum:
                            best_sum = srb
                            best_radii = rb.copy()
                        else:
                            best_centers[j] = oldb
                    else:
                        best_centers[j] = oldb

        return centers, radii, cur_sum

    # Diverse initializations
    inits = []

    # 1. near-grid spread to encourage uniformity
    grid_r, grid_c = 5, 6
    pts = []
    for i in range(grid_r):
        for j in range(grid_c):
            pts.append([(j + 0.5) / grid_c, (i + 0.5) / grid_r])
            if len(pts) == n:
                break
        if len(pts) == n:
            break
    inits.append(np.array(pts, dtype=float))

    # 2. random samples
    for _ in range(4):
        inits.append(random_center(n))

    # 3. biased toward center to avoid crowding walls early
    for _ in range(2):
        c = random_center(n)
        c = 0.9 * c + 0.05
        inits.append(clamp_centers(c))

    # 4. hierarchical local refinement seeds: small perturbations around grid
    base = inits[0]
    for k in range(3):
        pert = base + np.random.normal(scale=0.04, size=base.shape)
        inits.append(clamp_centers(pert))

    # Run attempts
    for init in inits:
        init = clamp_centers(init.copy())
        if init.shape != (n, 2):
            continue
        c, r, s = attempt(init, iter_limit=7000)
        if r is None:
            continue
        cur_sum = float(np.sum(r))
        if cur_sum > best_sum:
            best_sum = cur_sum
            best_centers = c.copy()
            best_radii = r.copy()

    # Fallback: if nothing found feasible, return a safe grid-like packing
    if best_centers is None:
        best_centers = np.array([[ (k % 6) / 5.0, (k // 6) / 5.0 ] for k in range(n)], dtype=float)
        # push inside a bit
        best_centers = clamp_centers(best_centers)
        best_radii = compute_radii_ordered(best_centers)
        if best_radii is None:
            best_radii = np.full(n, 0.01)

    final_radii = compute_radii_ordered(best_centers)
    if final_radii is None:
        final_radii = best_radii

    return best_centers, final_radii
