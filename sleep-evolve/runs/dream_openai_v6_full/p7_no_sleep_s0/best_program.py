def construct():
    # Deterministic, time-bounded, diversified greedy search for a small set B
    import time
    import numpy as np

    MAX_ELEMENTS = 2000
    TIME_LIMIT = 2.0  # seconds
    RNG_SEED = 20240614

    rng = np.random.default_rng(RNG_SEED)

    def compute_diffs(Bset):
        if not Bset:
            return np.zeros(0, dtype=np.int64)
        arr = np.array(sorted(Bset), dtype=np.int64)
        if arr.size <= 1:
            return np.zeros(0, dtype=np.int64)
        diff = arr[:, None] - arr[None, :]
        pos = diff[diff > 0]
        if pos.size == 0:
            return np.zeros(0, dtype=np.int64)
        return np.unique(pos)

    def largest_n(diffs):
        if diffs.size == 0:
            return 0
        ds = np.sort(diffs)
        n = 0
        i = 0
        m = ds.size
        while i < m and ds[i] == n + 1:
            n += 1
            i += 1
        return int(n)

    def candidate_pool_from(Bset, current_max):
        pool = set()
        base = [1, 2, 3, 4, 5, 7, 11, 13, 17]
        for d in base:
            pool.add(current_max + d)
        cap = min(80, max(0, int(current_max)) + 25)
        for b in range(0, cap):
            for d in base:
                pool.add(b + d)
        for _ in range(25):
            anchor = int(rng.integers(0, max(1, int(current_max) + 1)))
            pool.add(anchor + int(rng.integers(1, 9)))
        return [x for x in pool]

    def evaluate_gain(Bset, x, current_diffs_set, horizon_cap=260):
        newB = sorted(set(Bset) | {x})
        nb = np.array(newB, dtype=np.int64)
        diff2 = nb[:, None] - nb[None, :]
        new_diffs = np.unique(diff2[diff2 > 0])
        new_set = set(int(v) for v in new_diffs)
        current_set = set(int(v) for v in current_diffs_set)

        if new_diffs.size == 0:
            return 0
        horizon = int(min(horizon_cap, np.max(new_diffs)))
        gain = 0
        for k in range(1, horizon + 1):
            if (k not in current_set) and (k in new_set):
                gain += 1
        return gain

    best_overall = None  # (score, Bbest, nbest)
    best_score = -np.inf

    start_time = time.time()
    restarts = 0

    # Pre-seed: try small canonical B
    canonical_start = [0, 1, 3, 6, 10]
    initial_B = sorted(set([0] + canonical_start))
    diffs0 = compute_diffs(initial_B)
    n0 = largest_n(diffs0)
    score0 = - (len(initial_B) ** 2) / max(1, n0)
    best_overall = (score0, initial_B, n0)
    best_score = score0

    while (time.time() - start_time) < TIME_LIMIT:
        restarts += 1
        # Start from a small base to diversify
        B = {0}
        # random extra seed to diversify
        for s in [2, 4, 8]:
            if rng.random() < 0.5:
                B.add(int(rng.integers(1, 20) + s))

        local_start = time.time()
        while (len(B) < MAX_ELEMENTS) and ((time.time() - local_start) < TIME_LIMIT * 0.92):
            diffs = compute_diffs(B)
            n = largest_n(diffs)

            if n >= 260:
                break

            current_max = max(B) if B else 0
            cands = candidate_pool_from(B, current_max)
            if not cands:
                cands = [current_max + 1]

            rng.shuffle(cands)

            best_cand = None
            best_gain = -1

            for x in cands:
                if x in B or x < 0:
                    continue
                gain = evaluate_gain(B, x, diffs)
                if gain > best_gain:
                    best_gain = gain
                    best_cand = x

            if best_cand is None:
                t = 0
                while t in B:
                    t += 1
                best_cand = t

            B.add(best_cand)

            diffs = compute_diffs(B)
            if diffs.size:
                n = largest_n(diffs)
                if n >= 260:
                    break

        # finalize this run
        B.add(0)
        diffs = compute_diffs(B)
        n = largest_n(diffs)
        if n == 0:
            score = -np.inf
        else:
            score = - (len(B) ** 2) / n

        if (best_overall is None) or (score > best_overall[0]):
            best_overall = (score, sorted(B), n)
            best_score = score

        if best_overall and best_overall[2] >= 320 and len(best_overall[1]) <= 70:
            break

    if best_overall is None:
        return [0]

    _, Bbest, _ = best_overall
    Bbest = sorted(set(Bbest))
    if 0 not in Bbest:
        Bbest = [0] + Bbest
    if len(Bbest) > 2000:
        Bbest = Bbest[:2000]
    return Bbest
