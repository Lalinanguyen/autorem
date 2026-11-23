import numpy as np
import math

def compute_radii(centers):
    n = centers.shape[0]
    # distance to walls
    r_border = np.minimum.reduce([
        centers[:, 0], 1.0 - centers[:, 0],
        centers[:, 1], 1.0 - centers[:, 1]
    ])
    # pairwise distances
    dx = centers[:, None, 0] - centers[None, :, 0]
    dy = centers[:, None, 1] - centers[None, :, 1]
    dist = np.hypot(dx, dy)
    np.fill_diagonal(dist, np.inf)
    r_neigh = dist.min(axis=1) / 2.0
    radii = np.minimum(r_border, r_neigh)
    return radii

def hex_seed(n, pad=0.045, spacing=0.108):
    xs = []
    ys = []
    y = pad
    row = 0
    while len(xs) < n and y <= 1.0 - pad:
        dx = spacing * (0.5 if row % 2 else 0.0)
        x = pad + dx
        while len(xs) < n and x <= 1.0 - pad:
            xs.append(x)
            ys.append(y)
            x += spacing
        y += spacing * 0.866
        row += 1
    centers = np.column_stack([np.array(xs), np.array(ys)])
    if centers.shape[0] > n:
        centers = centers[:n]
    if centers.shape[0] < n:
        extra = n - centers.shape[0]
        extra_pts = np.clip(np.random.rand(extra, 2) * (1.0 - 2*pad) + pad, pad, 1.0 - pad)
        centers = np.vstack([centers, extra_pts])
    return centers

def jitter_seed(base, rng, jitter=0.04):
    cand = base.copy()
    cand += rng.normal(scale=jitter, size=cand.shape)
    return np.clip(cand, 0.0, 1.0)

def perturb(centers, scale, rng):
    cand = centers.copy()
    idx = rng.integers(0, centers.shape[0])
    delta = rng.normal(scale=scale, size=2)
    cand[idx, :] = np.clip(cand[idx, :] + delta, 0.0, 1.0)
    return cand

def place_sequential(n, rng, pad=0.04, attempts=800):
    pts = []
    for _ in range(n):
        placed = False
        for _ in range(attempts):
            cand = rng.random(2) * (1.0 - 2*pad) + pad
            if any(np.linalg.norm(cand - p) < 2e-9 for p in pts):
                continue
            ok = True
            for p in pts:
                if np.linalg.norm(cand - p) < 0.08:
                    ok = False
                    break
            if ok:
                pts.append(cand)
                placed = True
                break
        if not placed:
            # fallback: place randomly within bounds
            pts.append(rng.random(2) * (1.0 - 2*pad) + pad)
    pts = np.array(pts)
    if pts.shape[0] < n:
        extra = n - pts.shape[0]
        extra_pts = rng.random((extra, 2)) * (1.0 - 2*pad) + pad
        pts = np.vstack([pts, extra_pts])
    return pts

def construct():
    rng = np.random.default_rng(seed=987654)

    n = 26
    seeds = []

    # Seed families: hex lattice cores with jitter, plus a random spread seed
    seeds.append(hex_seed(n, pad=0.045, spacing=0.108))
    for _ in range(4):
        base = hex_seed(n, pad=0.045, spacing=0.108)
        seeds.append(jitter_seed(base, rng, jitter=0.04))
    seeds.append(place_sequential(n, rng, pad=0.04))
    # additional random spreads
    for _ in range(2):
        seeds.append(rng.random((n, 2)) * 0.92 + 0.04)

    best_centers = None
    best_radii = None
    best_sum = -1.0

    # Multi-start annealing/hill-climb
    for seed_idx, base_centers in enumerate(seeds):
        centers = base_centers.copy()
        radii = compute_radii(centers)
        cur_centers = centers
        cur_radii = radii
        cur_sum = float(np.sum(cur_radii))

        # simple temperature schedule
        T0 = 0.25
        Tf = 0.0003
        steps = 18000
        for t in range(steps):
            frac = t / (steps - 1)
            T = T0 * (Tf / max(T0, 1e-12))**frac
            scale = max(0.012, 0.18 * (1.0 - frac))

            cand_centers = perturb(cur_centers, scale=scale, rng=rng)
            cand_radii = compute_radii(cand_centers)
            cand_sum = float(np.sum(cand_radii))

            if cand_sum > cur_sum + 1e-12:
                cur_centers = cand_centers
                cur_radii = cand_radii
                cur_sum = cand_sum
            else:
                delta = cand_sum - cur_sum
                if delta < 0:
                    prob = math.exp(delta / max(T, 1e-12))
                    if rng.random() < prob:
                        cur_centers = cand_centers
                        cur_radii = cand_radii
                        cur_sum = cand_sum

        if cur_sum > best_sum:
            best_sum = cur_sum
            best_centers = cur_centers
            best_radii = cur_radii

    # Final refinement pass on best_centers
    centers = best_centers.copy()
    radii = compute_radii(centers)
    cur_centers = centers
    cur_radii = radii
    cur_sum = float(np.sum(cur_radii))

    # Focused local tweaks
    for _ in range(7000):
        i = rng.integers(0, n)
        cand = cur_centers.copy()
        delta = rng.normal(scale=0.025, size=2)
        cand[i] = np.clip(cand[i] + delta, 0.0, 1.0)
        cand_r = compute_radii(cand)
        cand_sum = float(np.sum(cand_r))
        if cand_sum > cur_sum + 1e-12:
            cur_centers = cand
            cur_radii = cand_r
            cur_sum = cand_sum

    final_centers = cur_centers
    final_radii = compute_radii(final_centers)

    # Safeguard non-negative radii
    final_radii = np.maximum(final_radii, 0.0)

    return final_centers, final_radii
