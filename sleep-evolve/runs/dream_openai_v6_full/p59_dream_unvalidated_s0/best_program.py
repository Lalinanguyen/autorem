import random
import time

def construct():
    # Grid 0..63
    n = 64
    points = [(x, y) for x in range(n) for y in range(n)]
    total = len(points)

    # Time budget
    start_time = time.time()
    time_limit = 3.5  # seconds

    # Seed for reproducibility
    random_seed = 20241123
    random.seed(random_seed)

    best_pts = []
    best_size = 0

    # Helper: can we add candidate p given current selected pts and per-point forbiddens?
    def can_add(p, selected, forbiddens):
        # Compare distances to all existing points; must be all distinct
        d2s = []
        for i, q in enumerate(selected):
            dx = p[0] - q[0]
            dy = p[1] - q[1]
            d2 = dx*dx + dy*dy
            if d2 in d2s:  # duplicate distance to two existing points -> would violate "all distances distinct"
                return None
            d2s.append(d2)
            if d2 in forbiddens[i]:
                # distance to this existing point is forbidden (already used with some previous point)
                return None
        return d2s  # list of distances to existing points

    def greedy_build(order):
        selected = []
        forbiddens = []  # for each selected point i, forbiddens[i] is a set of d2 values (distances to future points)
        for idx in order:
            p = points[idx]
            if not selected:
                # Always can take first point
                selected.append(p)
                forbiddens.append(set())
                continue
            d2s = can_add(p, selected, forbiddens)
            if d2s is None:
                continue
            # Accept p: update forbiddens
            for i, d2 in enumerate(d2s):
                forbiddens[i].add(d2)
            forbiddens.append(set(d2s))
            selected.append(p)
        return selected

    # A few seed ordering strategies
    def biased_order():
        cx = 31.5
        cy = 31.5
        arr = list(range(total))
        scores = []
        for idx in arr:
            x, y = points[idx]
            dist2 = (x - cx)**2 + (y - cy)**2
            scores.append((dist2, random.random(), idx))
        scores.sort(reverse=True)
        return [t[2] for t in scores]

    def random_order():
        order = list(range(total))
        random.shuffle(order)
        return order

    # Main loop: multiple restarts
    while time.time() - start_time < time_limit:
        if random.random() < 0.5:
            order = random_order()
        else:
            order = biased_order()

        candidate = greedy_build(order)
        if len(candidate) > best_size:
            best_size = len(candidate)
            best_pts = candidate
            if best_size >= 120:
                break

        # Lightweight pruning attempt: try to prune from best_pts if it helps
        if best_pts:
            # Try flipping a few random points out to allow adding others
            current = best_pts[:]
            current_forb = []
            for i in range(len(current)):
                current_forb.append(set())

            # Create a quick local re-check by attempting to remove a small random subset
            remove_count = max(1, int(len(current) * 0.15))
            to_try_remove = random.sample(range(len(current)), remove_count)
            new_selected = [p for idx, p in enumerate(current) if idx not in set(to_try_remove)]
            # Recompute forbiddens for new_selected
            new_forb = [set() for _ in range(len(new_selected))]
            ok_all = True
            for i in range(len(new_selected)):
                for j in range(i):
                    dx = new_selected[i][0] - new_selected[j][0]
                    dy = new_selected[i][1] - new_selected[j][1]
                    d2 = dx*dx + dy*dy
                    # enforce distinct distances across existing in new_selected
                    if d2 in new_forb[j]:
                        ok_all = False
                        break
                    new_forb[i].add(d2)
                if not ok_all:
                    break
            if ok_all and len(new_selected) > best_size:
                best_pts = new_selected
                best_size = len(new_selected)
                if best_size >= 120:
                    break

    # Finalize: return best found as list of [x, y]
    return [list(p) for p in best_pts]
