import random
import time
import math

def construct():
    # Grid size
    n = 64
    coords = [(x, y) for x in range(n) for y in range(n)]
    N = n * n

    # Distance helpers
    def dist2(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return dx * dx + dy * dy

    # Validity check when trying to add p to current set pts.
    # seen_by is a list of sets: seen_by[i] contains squared distances from point i to all currently chosen points.
    def try_add(p, pts, seen_by):
        d2s = set()
        for q in pts:
            d2 = dist2(p, q)
            # Distances from p to existing q must be all distinct
            if d2 in d2s:
                return None
            # Distances from p to q must not collide with any distance already seen from q to others
            if d2 in seen_by[q[0] * n + q[1]]:
                return None
            d2s.add(d2)
        return d2s

    # Commit addition: update seen_by for p and all existing pts
    def commit_add(p, pts, seen_by, d2s):
        for q in pts:
            d2 = dist2(p, q)
            seen_by[p[0] * n + p[1]].add(d2)
            seen_by[q[0] * n + q[1]].add(d2)
        pts.append(p)

    # Run a single greedy pass over a given order of indices
    def run_once(order):
        pts = []
        seen_by = [set() for _ in range(N)]
        for idx in order:
            p = (idx // n, idx % n)
            d2s = try_add(p, pts, seen_by)
            if d2s is not None:
                commit_add(p, pts, seen_by, d2s)
        return pts

    # Deterministic seeds: start from a checkerboard-like coarse seed
    seed_pts = [(x, y) for x in range(0, n, 2) for y in range(0, n, 2)]
    best_pts = []
    best_size = 0

    # 1) seed-based greedy to bootstrap
    order_seed = list(range(n * n))
    random.shuffle(order_seed)
    baseline = run_once(order_seed)
    if len(baseline) > best_size:
        best_pts = baseline
        best_size = len(best_pts)

    # Also try a few deterministic seeds to diversify starting geometry
    for sx in range(0, n, 4):
        for sy in range(0, n, 4):
            order = []
            # start with seed subset, then random rest
            seed_idx = [(sx + dx) % n * n + (sy + dy) % n for dx in (0,1) for dy in (0,1)]
            # Build an order that prioritizes seed points
            rest = [i for i in range(N) if i not in seed_idx]
            random.shuffle(rest)
            order = seed_idx + rest
            cand = run_once(order)
            if len(cand) > best_size:
                best_pts = cand
                best_size = len(best_pts)

    # 2) randomized restarts with limited time
    time_budget = 0.9  # seconds
    start_time = time.time()

    while time.time() - start_time < time_budget:
        idxs = list(range(N))
        random.shuffle(idxs)
        cand = run_once(idxs)
        if len(cand) > best_size:
            best_pts = cand
            best_size = len(best_pts)

    # Final validation to ensure correctness (no isosceles apexes)
    # Build quick verification
    def is_valid_set(pts):
        m = len(pts)
        for i in range(m):
            ai = pts[i]
            d2s = []
            seen = set()
            for j in range(m):
                if i == j: continue
                dj = dist2(ai, pts[j])
                if dj in seen:
                    return False
                seen.add(dj)
        # Also verify pairwise distance uniqueness from each point to others
        for i in range(m):
            dset = set()
            for j in range(m):
                if i == j: continue
                d2 = dist2(pts[i], pts[j])
                if d2 in dset:
                    return False
                dset.add(d2)
        return True

    if best_pts and not is_valid_set(best_pts):
        # As a safety net, fall back to a small valid subset (basic greedy reconstruction)
        best_pts = []
        seen_by = [set() for _ in range(N)]
        order = list(range(N))
        random.shuffle(order)
        for idx in order:
            p = (idx // n, idx % n)
            d2s = try_add(p, best_pts, seen_by)
            if d2s is not None:
                commit_add(p, best_pts, seen_by, d2s)

    return [list(p) for p in best_pts]
