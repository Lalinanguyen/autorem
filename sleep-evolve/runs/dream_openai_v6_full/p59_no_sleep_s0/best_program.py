import time
import random

def construct():
    """
    Improved randomized-greedy with reinforced local search for the isosceles constraint.
    - Maintains per-point distance sets to ensure all distances from a point to existing points are unique.
    - Adds a two-stage local search: small perturbations and periodic larger removals with reinsertion.
    - Uses a modest time budget to balance exploration and convergence toward 112+ points.
    """
    n = 64
    N = n * n
    all_points = [(x, y) for x in range(n) for y in range(n)]

    best_pts = []
    best_size = 0

    time_budget = 4.0
    t0 = time.time()

    def idx(p):
        x, y = p
        return x * n + y

    def can_add(p, pts, in_set, dist_sets):
        x, y = p
        i = idx(p)
        if in_set[i]:
            return False, None
        d2s = set()
        for (xq, yq) in pts:
            j = idx((xq, yq))
            d2 = (x - xq) * (x - xq) + (y - yq) * (y - yq)
            if d2 in d2s:
                return False, None
            # If this distance duplicates an existing distance from the candidate to any current point,
            # that would make that current point have two equal distances.
            if d2 in dist_sets[j]:
                return False, None
            d2s.add(d2)
        return True, d2s

    def commit(p, d2s, pts, in_set, dist_sets):
        x, y = p
        i = idx((x, y))
        in_set[i] = True
        pts.append((x, y))
        # update distance records for both new point and existing points
        for (xq, yq) in pts:
            if (xq, yq) == (x, y):
                continue
            j = idx((xq, yq))
            d2 = (x - xq) * (x - xq) + (y - yq) * (y - yq)
            dist_sets[i].add(d2)
            dist_sets[j].add(d2)

    def remove_point(p, pts, in_set, dist_sets):
        x, y = p
        i = idx((x, y))
        if not in_set[i]:
            return
        in_set[i] = False
        # Remove contributions from dist_sets
        for (xq, yq) in pts:
            if (xq, yq) == (x, y):
                continue
            j = idx((xq, yq))
            d2 = (x - xq) * (x - xq) + (y - yq) * (y - yq)
            dist_sets[i].discard(d2)
            dist_sets[j].discard(d2)
        # remove from pts
        for k in range(len(pts)):
            if pts[k] == (x, y):
                pts.pop(k)
                break

    def greedy_build(seed):
        random.seed(seed)
        pts = []
        in_set = [False] * N
        dist_sets = [set() for _ in range(N)]

        order = all_points[:]
        random.shuffle(order)

        # Phase 1: greedy insertion
        for p in order:
            ok, d2s = can_add(p, pts, in_set, dist_sets)
            if ok:
                commit(p, d2s, pts, in_set, dist_sets)

        # Phase 2: small perturbation - remove a small random subset
        if len(pts) >= 4:
            k = max(1, len(pts) // 6)
            rem_indices = sorted(random.sample(range(len(pts)), k), reverse=True)
            removed = []
            for idxp in rem_indices:
                px, py = pts[idxp]
                removed.append((px, py))
                remove_point((px, py), pts, in_set, dist_sets)
        else:
            removed = []

        # Phase 3: try reinserting removed points (and any remaining non-members)
        remaining = [p for p in all_points if not in_set[idx(p)]]
        random.shuffle(remaining)
        for p in remaining:
            ok, d2s = can_add(p, pts, in_set, dist_sets)
            if ok:
                commit(p, d2s, pts, in_set, dist_sets)

        # Phase 4: attempt to reinsert previously removed points
        for p in removed:
            ok, d2s = can_add(p, pts, in_set, dist_sets)
            if ok:
                commit(p, d2s, pts, in_set, dist_sets)

        return pts

    # Phase 5: iterative restarts with occasional larger perturbations
    seeds = [101741527, 123457, 9876541, 424242, 7777777, 1928374] + [7*i ^ 0x9e3779b9 for i in range(20)]
    best_in_time = 0

    while time.time() - t0 < time_budget:
        seed = random.choice(seeds)
        pts = greedy_build(seed)
        if len(pts) > best_size:
            best_size = len(pts)
            best_pts = pts
            best_in_time = time.time() - t0
            if best_size >= 112:
                break

        # Occasional more aggressive perturbation restart
        if random.random() < 0.18:
            pts = greedy_build(seed ^ 0x9e3779b9)
            if len(pts) > best_size:
                best_size = len(pts)
                best_pts = pts
                best_in_time = time.time() - t0
                if best_size >= 112:
                    break

    # Final cleanup: deduplicate and ensure bounds
    seen = set()
    final = []
    for p in best_pts:
        x, y = p
        if 0 <= x < n and 0 <= y < n:
            if (x, y) not in seen:
                seen.add((x, y))
                final.append([x, y])

    return final
