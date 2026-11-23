import numpy as np
import time

def construct():
    """
    Enhanced randomized-search with richer seeding and stronger post-placement refinement.
    - Diversified deterministic seeds + large-scale restarts
    - Expanded candidate sampling around existing centers as well as global sampling
    - Aggressive multi-pass refinement to push radii while preserving feasibility
    - Final clamp to enforce feasible (non-overlapping) packing
    """
    n = 26
    best_centers = None
    best_radii = None
    best_score = -1.0

    rng = np.random.default_rng(987654321)

    def r_wall(x, y):
        # radius limited by distance to walls
        return min(x, 1.0 - x, y, 1.0 - y)

    TIME_LIMIT = 9.0
    t0 = time.time()

    RESTARTS = 260
    CAND_PER_STEP = 700 if False else 700  # tuned for broader search

    # Expanded seeds to improve coverage
    seed_positions = [
        (0.10, 0.10), (0.90, 0.10), (0.10, 0.90), (0.90, 0.90),
        (0.50, 0.50), (0.50, 0.20), (0.20, 0.50), (0.80, 0.50),
        (0.33, 0.33), (0.67, 0.33), (0.33, 0.67), (0.67, 0.67),
        (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75),
        (0.40, 0.60), (0.60, 0.40), (0.15, 0.60), (0.60, 0.15),
        (0.35, 0.15), (0.15, 0.35), (0.85, 0.35), (0.35, 0.85),
        (0.70, 0.70)
    ]

    def run_one(seed):
        centers = []
        radii = []

        rand = np.random.RandomState(seed)

        # Seed placement: place a subset of seeds with moderate radii
        for (sx, sy) in seed_positions[:min(12, n)]:
            r = min(r_wall(sx, sy), 0.14)
            ok = True
            for (cx, cy), ri in zip(centers, radii):
                if np.hypot(sx - cx, sy - cy) < ri + r - 1e-12:
                    ok = False
                    break
            if ok:
                centers.append(np.array([sx, sy]))
                radii.append(float(r))
                if len(radii) == n:
                    break

        # Growth phase: broaden sampling to fill remaining circles
        for k in range(len(radii), n):
            best_local_r = -1.0
            best_local_center = None

            for _ in range(CAND_PER_STEP):
                # mix: either around a random existing center or global
                if centers and rand.rand() < 0.6:
                    idx = rand.randint(0, len(centers))
                    ref = centers[idx]
                    angle = rand.rand() * 2 * np.pi
                    dist = rand.rand() * 0.9
                    cx = ref[0] + dist * np.cos(angle)
                    cy = ref[1] + dist * np.sin(angle)
                else:
                    cx = rand.rand()
                    cy = rand.rand()

                cx = min(max(cx, 0.0), 1.0)
                cy = min(max(cy, 0.0), 1.0)

                rw = r_wall(cx, cy)
                if rw <= 1e-12:
                    continue
                rmax = rw
                for (cx_i, cy_i), ri in zip(centers, radii):
                    d = np.hypot(cx - cx_i, cy - cy_i)
                    rmax = min(rmax, d - ri)
                if rmax > best_local_r:
                    best_local_r = rmax
                    best_local_center = (cx, cy)

            if best_local_center is None or best_local_r <= 1e-12:
                break

            centers.append(np.array(best_local_center))
            radii.append(float(best_local_r))

        # Post-process: multi-pass refinement to improve radii
        if len(radii) == n:
            centers_arr = np.array(centers, dtype=float)
            radii_arr = np.array(radii, dtype=float)

            # Several refinement sweeps with larger perturbations
            for sweep in range(180):
                i = rand.randint(0, n)
                angle = rand.rand() * 2 * np.pi
                mag = (rand.rand() - 0.5) * 0.12  # +/-0.06
                dx = mag * np.cos(angle)
                dy = mag * np.sin(angle)
                cx = centers_arr[i, 0] + dx
                cy = centers_arr[i, 1] + dy
                cx = min(max(cx, 0.0), 1.0)
                cy = min(max(cy, 0.0), 1.0)

                rwall = r_wall(cx, cy)
                rnew = rwall
                for j in range(n):
                    if j == i:
                        continue
                    d = np.hypot(cx - centers_arr[j, 0], cy - centers_arr[j, 1])
                    rnew = min(rnew, d - radii_arr[j])
                if rnew > radii_arr[i] + 1e-12:
                    centers_arr[i] = np.array([cx, cy])
                    radii_arr[i] = float(rnew)

            score = float(radii_arr.sum())
            return centers_arr, radii_arr, score
        else:
            return None

    for rep in range(RESTARTS):
        if time.time() - t0 > TIME_LIMIT:
            break
        seed = int(rng.integers(1 << 30))
        res = run_one(seed)
        if res is None:
            continue
        centers_arr, radii_arr, score = res

        if score > best_score:
            best_score = score
            best_centers = centers_arr
            best_radii = radii_arr

            # Early stop if a strong packing is found
            if score >= 2.66:
                break

    # Fallback deterministic layout if nothing found
    if best_centers is None:
        cols, rows = 5, 6
        r = min(1.0 / max(cols, rows), 0.18)
        centers = []
        radii = []
        for k in range(n):
            i, j = divmod(k, cols)
            cx = (j + 0.5) / cols
            cy = (i + 0.5) / rows
            rr = min(r, r_wall(cx, cy))
            centers.append([cx, cy])
            radii.append(float(rr))
        best_centers = np.array(centers)
        best_radii = np.array(radii)

    best_centers = np.asarray(best_centers, dtype=float)
    best_radii = np.asarray(best_radii, dtype=float)

    # Final feasibility clamp
    for i in range(n):
        r = best_radii[i]
        x, y = best_centers[i]
        r = min(r, r_wall(x, y))
        best_radii[i] = max(r, 0.0)

    return best_centers, best_radii
