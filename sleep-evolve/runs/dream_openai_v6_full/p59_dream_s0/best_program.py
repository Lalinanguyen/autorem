import random
import time

def construct():
    """
    Build a large subset of the 64x64 grid with the property:
    for every chosen point p, the distances to all other chosen points are pairwise distinct.

    Improvements over previous attempts:
    - Stronger feasibility check with per-point distance tracking
    - Time-budgeted multi-start search with both full and prefix-preserving shuffles
    - Explicit validity guarantees on the final output shape
    """
    n = 64
    all_pts = [(x, y) for x in range(n) for y in range(n)]

    TOTAL_TIME_SECONDS = 3.2
    t_start = time.time()

    def dist2(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return dx*dx + dy*dy

    # Data structures kept while building a candidate set
    best_pts = []
    best_size = 0

    def can_add(p, pts, dist_sets):
        # Distances from p to existing pts must be all distinct
        dists_to_p = []
        for q in pts:
            d = dist2(p, q)
            if d in dists_to_p:
                return False
            dists_to_p.append(d)
            # Distance to p must be new for q as well
            if d in dist_sets.get(q, set()):
                return False
        return True

    def add_point(p, pts, dist_sets):
        # Update structures after adding p
        new_dists = [dist2(p, q) for q in pts]
        for i, q in enumerate(pts):
            d = new_dists[i]
            dist_sets.setdefault(q, set()).add(d)
        dist_sets[p] = set(new_dists)
        pts.append(p)

    def attempt_order(order):
        pts = []
        dist_sets = {}
        for p in order:
            if can_add(p, pts, dist_sets):
                add_point(p, pts, dist_sets)
        return pts

    # Seed randomness consistently per run
    base_seed = int(time.time() * 1000) ^ 0x9e3779b97f4a7c15
    random.seed(base_seed)

    # Use a base "core" order that tends to place diverse points early
    base_order = all_pts[:]
    random.shuffle(base_order)

    # Helper to build a candidate from a given ordering strategy
    def build_from_order(order):
        return attempt_order(order)

    # Main loop with time budget
    while time.time() - t_start < TOTAL_TIME_SECONDS:
        order = base_order[:]
        r = random.random()

        # Strategy mix: full randomization vs prefix-stable shuffles
        if r < 0.65:
            random.shuffle(order)
        else:
            head_len = max(1, int(0.25 * len(order)))
            head = order[:head_len]
            tail = order[head_len:]
            random.shuffle(tail)
            order = head + tail

        pts = build_from_order(order)

        if len(pts) > best_size:
            best_size = len(pts)
            best_pts = pts
            if best_size >= 112:
                break

        # Local perturbation: extend from a warped subset of best_pts
        if best_pts:
            subset = best_pts[:]
            random.shuffle(subset)
            cut = max(1, int(0.25 * len(subset)))
            partial = subset[:-cut]
            if partial:
                order2 = partial + all_pts
                pts2 = build_from_order(order2)
                if len(pts2) > best_size:
                    best_size = len(pts2)
                    best_pts = pts2
                    if best_size >= 112:
                        break

    if not best_pts:
        return []

    # Normalize to required output: list of [x, y] ints
    return [list(p) for p in best_pts]
