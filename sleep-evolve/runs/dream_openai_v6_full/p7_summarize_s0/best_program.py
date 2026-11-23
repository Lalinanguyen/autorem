import random
import time
from itertools import combinations

def compute_n(B):
    # compute the largest n such that 1..n are all differences of pairs in B
    s = set()
    B = sorted(set(B))
    for i in range(len(B)):
        for j in range(i+1, len(B)):
            s.add(B[j] - B[i])
    n = 0
    while (n + 1) in s:
        n += 1
    if n == 0:
        return 0
    return n

def all_differences(B):
    s = set()
    B = sorted(set(B))
    for i in range(len(B)):
        for j in range(i+1, len(B)):
            s.add(B[j] - B[i])
    return s

def contrast_score(B, n=None):
    B = sorted(set(B))
    if n is None:
        n = compute_n(B)
    if n <= 0:
        return -10.0  # invalid
    return - (len(B) ** 2) / float(n)

def construct():
    """
    Attempt to build a small set B (size 4-6) that covers 1..n with maximal n.
    Uses a randomized greedy exploration, favoring early small elements (to keep max diff reasonable)
    and occasional restarts to escape local optima.
    """
    # Time budget to keep within typical evaluation limits
    TIME_BUDGET = 0.95  # seconds
    tstart = time.time()
    best_B = None
    best_n = 0
    best_score = -1e9  # we want higher (less negative)

    # Start with a few deterministic seeds that historically perform well (0 in B helps diffs)
    seeds = [
        [0, 1, 3, 4],   # known good 4-element set
        [0, 1, 4, 6],   # another balanced 4-set
        [0, 1, 3, 6],   # varied
        [0, 2, 3, 5],   # alternative structure
        [0, 1, 5, 10],  # extended max diff potential
    ]

    # helper to try building from a seed
    def hill_from(seed, max_size=6):
        current = sorted(set(seed))
        current = current[:max_size]
        best_local = list(current)
        best_local_n = compute_n(best_local)
        best_local_score = contrast_score(best_local, best_local_n)
        elements_pool = list(range(0, 25))  # limit search space to keep it fast
        random.shuffle(elements_pool)
        # greedy additives
        while len(current) < max_size:
            improved = False
            for cand in elements_pool:
                if cand in current:
                    continue
                trial = sorted(set(current + [cand]))
                n = compute_n(trial)
                if n > best_local_n:
                    best_local_n = n
                    best_local = trial
                    best_local_score = contrast_score(best_local, best_local_n)
                    current = best_local
                    improved = True
                    break
            if not improved:
                # try small random perturbation
                cand = random.choice([x for x in range(0, 25) if x not in current])
                trial = sorted(set(current + [cand]))
                n = compute_n(trial)
                score = contrast_score(trial, n)
                if score > best_local_score:
                    best_local = trial
                    best_local_n = n
                    best_local_score = score
                    current = best_local
        return best_local, best_local_n

    # Explore with seeds and random restarts
    # initialize with best of seeds
    for seed in seeds:
        B, n = hill_from(seed, max_size=6)
        sc = contrast_score(B, n)
        if sc > best_score:
            best_score = sc
            best_B = B
            best_n = n

    # Randomized restarts within time budget
    while time.time() - tstart < TIME_BUDGET:
        # random seed: include 0 always to maximize diff opportunities
        size = random.choice([4, 5, 6])
        B = {0}
        while len(B) < size:
            B.add(random.randint(0, 20))
        B = sorted(B)
        # refine with hill climbing from this random seed
        B_ref, n_ref = hill_from(B, max_size=size)
        sc = contrast_score(B_ref, n_ref)
        if sc > best_score:
            best_score = sc
            best_B = B_ref
            best_n = n_ref

    # Final dedup and sort, ensure 0 present
    if best_B is None or compute_n(best_B) == 0:
        best_B = [0, 1, 3, 4]
        best_n = compute_n(best_B)
        best_score = contrast_score(best_B, best_n)

    # Ensure non-negative integers and deduplicated, cap size
    B_final = sorted(set(best_B))
    if B_final and B_final[0] != 0:
        B_final = [0] + B_final
    if len(B_final) > 2000:
        B_final = B_final[:2000]
    return B_final
