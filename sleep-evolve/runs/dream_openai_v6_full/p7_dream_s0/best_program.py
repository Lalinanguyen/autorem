import numpy as np

def _max_consecutive_coverage(B):
    if len(B) < 2:
        return 0
    B = np.asarray(sorted(np.unique(B)), dtype=int)
    # all pairwise absolute differences
    diffs = np.abs(B[:, None] - B[None, :])
    D = np.unique(diffs)
    D = D[D > 0]
    if D.size == 0:
        return 0
    max_d = int(D.max())
    present = np.zeros(max_d + 1, dtype=bool)
    present[D.astype(int)] = True
    n = 0
    while n + 1 < present.size and present[n + 1]:
        n += 1
    return int(n)

def _best_candidate_for_growth(B, candidates, base_n):
    Bset = set(B)
    best_gain = -1e9
    best = None
    for c in candidates:
        if c in Bset:
            continue
        newB = np.array(sorted(list(Bset) + [c]), dtype=int)
        n = _max_consecutive_coverage(newB)
        gain = n - base_n
        if gain > best_gain:
            best_gain = gain
            best = c
    return best, best_gain

def construct():
    """
    Deterministic, small-pool greedy growth with a size-aware acceptance rule:
    - seed with [0,1] to guarantee 1 as a difference
    - iteratively try a small, well-scoped pool of candidates around current max
      and accept the candidate if it increases the consecutive coverage n by at least 1
      while keeping growth modest to avoid bloating |B|.
    - if no improving candidate, try a tiny augmentation; stop when no gain is possible
    - keep at most 2000 elements; ensure 0 is present
    """
    max_size = 2000
    B = [0, 1]
    B = sorted(set(B))
    base_n = _max_consecutive_coverage(B)
    rng = np.random.default_rng(seed=20241123)

    # Deterministic exploration window; we still sample a fixed small pool for diversity
    for iteration in range(1200):
        current_max = int(max(B))
        window = max(int(current_max * 2) + 60, 120)

        candidates = []

        # Small, structured pools around current range
        near_zero = rng.integers(0, min(window, 400) + 1, size=6).tolist()
        mid_region = rng.integers(max(0, current_max - window // 4), window + 1, size=10).tolist()
        high_region = rng.integers(window // 2, window + 1, size=8).tolist()

        candidates.extend(near_zero)
        candidates.extend(mid_region)
        candidates.extend(high_region)

        # Deduplicate and filter non-negatives
        candidates = [int(x) for x in set(candidates) if int(x) >= 0]

        if not candidates:
            break

        best_candidate, gain = _best_candidate_for_growth(B, candidates, base_n)

        if gain <= 0 or best_candidate is None:
            # Safe small augmentation: try a single new point far from current set
            trial = int(rng.integers(0, window + 1))
            if trial in B:
                break
            newB = np.array(sorted(list(set(B) | {trial})), dtype=int)
            new_n = _max_consecutive_coverage(newB)
            if new_n <= base_n:
                break
            B.append(trial)
            B = sorted(set(B))
            base_n = new_n
            if len(B) >= max_size:
                break
            continue

        # Apply best candidate
        B.append(int(best_candidate))
        B = sorted(set(B))
        base_n = _max_consecutive_coverage(B)

        if len(B) >= max_size:
            break
        if base_n >= 150:  # strong coverage achieved, can stop early
            break

    # Ensure 0 is present
    if 0 not in B:
        B.append(0)
    B = sorted(set(B))
    if len(B) == 0:
        B = [0]
    return B
