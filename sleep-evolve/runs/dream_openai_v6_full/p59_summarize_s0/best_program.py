import time
import random

def _is_valid_set(pts):
    # For every point, distances to all other points must be pairwise distinct
    n = len(pts)
    if n <= 1:
        return True
    for i in range(n):
        d2set = set()
        xi, yi = pts[i]
        for j in range(n):
            if i == j:
                continue
            xj, yj = pts[j]
            dx = xi - xj
            dy = yi - yj
            d2 = dx*dx + dy*dy
            if d2 in d2set:
                return False
            d2set.add(d2)
    return True

def _greedy_build(order):
    pts = []          # selected points as (x,y)
    masks = []        # per existing point: set of distances to later-added points
    coords = []       # copy of selected coords

    for p in order:
        px, py = p
        if not pts:
            pts.append((px, py))
            coords.append((px, py))
            masks.append(set())
            continue

        # distances to existing
        d2s = []
        for (qx, qy) in coords:
            dx = px - qx
            dy = py - qy
            d2s.append(dx*dx + dy*dy)

        # 1) all d2s must be unique
        if len(set(d2s)) != len(d2s):
            continue

        # 2) for every existing i, d2s[i] must not already be in masks[i]
        ok = True
        for i, d2 in enumerate(d2s):
            if d2 in masks[i]:
                ok = False
                break
        if not ok:
            continue

        # Accept
        for i, d2 in enumerate(d2s):
            masks[i].add(d2)
        masks.append(set(d2s))
        coords.append((px, py))
        pts.append((px, py))

    return pts

def _center_order(n, all_pts):
    cx = (n - 1) / 2.0
    cy = (n - 1) / 2.0
    def key(p):
        x, y = p
        dman = abs(x - cx) + abs(y - cy)
        d2 = (x - cx)**2 + (y - cy)**2
        return (dman, d2)
    return sorted(all_pts, key=key)

def _corners_order(n, all_pts):
    cx = (n - 1) / 2.0
    cy = (n - 1) / 2.0
    corners = [(0,0),(0,n-1),(n-1,0),(n-1,n-1)]
    def key(p):
        x, y = p
        d_corner = min((x - cx2)**2 + (y - cy2)**2 for cx2, cy2 in corners)
        d_center = (x - cx)**2 + (y - cy)**2
        return (d_corner, d_center)
    return sorted(all_pts, key=key)

def _spiral_order(n, all_pts):
    cx = (n - 1) / 2.0
    cy = (n - 1) / 2.0
    order = []
    maxr = max(n-1, n-1)
    seen = set()
    for r in range(0, maxr + 1):
        x0, x1 = int(round(cx - r)), int(round(cx + r))
        y0, y1 = int(round(cy - r)), int(round(cy + r))
        # top row
        if 0 <= y0 < n:
            for x in range(max(0, x0), min(n, x1 + 1)):
                p = (x, y0)
                if p not in seen:
                    order.append(p)
                    seen.add(p)
        # right col
        if 0 <= x1 < n:
            for y in range(max(0, y0 + 1), min(n, y1 + 1)):
                p = (x1, y)
                if p not in seen:
                    order.append(p)
                    seen.add(p)
        # bottom row
        if 0 <= y1 < n:
            for x in range(min(n - 1, x1 - 1), max(-1, x0 - 1), -1):
                p = (x, y1)
                if p not in seen:
                    order.append(p)
                    seen.add(p)
        # left col
        if 0 <= x0 < n:
            for y in range(min(n - 1, y1 - 1), max(-1, y0 - 1), -1):
                p = (x0, y)
                if p not in seen:
                    order.append(p)
                    seen.add(p)
    # ensure full coverage if something missed
    if len(order) < len(all_pts):
        remain = [p for p in all_pts if p not in set(order)]
        order.extend(remain)
    return order

def construct():
    n = 64
    all_pts = [(x, y) for x in range(n) for y in range(n)]
    best_sol = []
    best_len = 0

    start_time = time.time()
    TIME_LIMIT = 6.0  # seconds, generous for Python

    # Deterministic seed orders
    seed_orders = [
        _center_order(n, all_pts),
        _corners_order(n, all_pts),
        _spiral_order(n, all_pts),
    ]

    # 1) Run deterministic seeds
    for order in seed_orders:
        sol = _greedy_build(order)
        if len(sol) > best_len and _is_valid_set(sol):
            best_sol = sol
            best_len = len(sol)

    # 2) Time-bounded randomized restarts with light perturbations
    pool = all_pts[:]
    random.shuffle(pool)

    def refill_after_removals(base_sol, pool_order):
        # Rebuild from a base with a fresh pool order
        return _greedy_build(pool_order if base_sol is None else base_sol + pool_order)

    while time.time() - start_time < TIME_LIMIT:
        # create a shuffled order with tiny deterministic perturbation
        random.shuffle(pool)

        # mild perturbation: small swaps to inject diversity
        order = pool[:]
        if len(order) > 500:
            step = max(1, len(order) // 12)
            for i in range(0, len(order) - step, step):
                if random.random() < 0.3:
                    j = i + random.randint(1, min(step, len(order) - 1 - i))
                    order[i], order[j] = order[j], order[i]

        sol = _greedy_build(order)
        if len(sol) > best_len and _is_valid_set(sol):
            best_sol = sol
            best_len = len(sol)

        # 3) Lightweight local repair: try removing 1-2 points from best and refill
        if best_len >= 2:
            base = best_sol[:]
            for remove_k in (1, 2):
                if best_len - remove_k <= 0:
                    continue
                if len(base) < remove_k:
                    continue
                idxs = random.sample(range(len(base)), remove_k)
                trial = [p for i, p in enumerate(base) if i not in idxs]
                # refill with a fresh shuffled order
                refill_order = pool[:]
                random.shuffle(refill_order)
                sol2 = _greedy_build(trial + refill_order)
                if len(sol2) > best_len and _is_valid_set(sol2):
                    best_sol = sol2
                    best_len = len(sol2)

    # Final safeguard: ensure validity, otherwise fall back to best-known valid subset
    if not _is_valid_set(best_sol):
        # Very small fallback: try from a clean start with center seed
        fallback = _greedy_build(_center_order(n, all_pts))
        if _is_valid_set(fallback) and len(fallback) > best_len:
            best_sol = fallback
            best_len = len(fallback)

    # Final formatting
    return [ [int(x), int(y)] for (x, y) in best_sol ]
