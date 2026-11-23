import itertools
import math

def compute_n(B):
    if not B:
        return 0
    Bset = set(B)
    # compute all positive differences
    diffs = set()
    for a in B:
        for b in B:
            if a != b:
                diffs.add(abs(a - b))
    if 1 not in diffs:
        return 0
    # find largest n such that 1..n are all present
    n = 1
    while n in diffs:
        n += 1
    return n - 1

def score_of(B):
    n = compute_n(B)
    if n <= 0:
        return float('-inf')
    return - (len(B) ** 2) / n

def construct():
    # deterministic seeds (initial bases)
    seeds = [
        [0, 1],
        [0, 2],  # alternative base
        [0, 3],  # another seed
        [0, 1, 2],  # richer base
        [0, 1, 4],
    ]
    max_elem_bound = 600  # cap to keep evaluation fast
    best_overall = []
    best_score = float('-inf')

    # helper: ensure 0 included
    def normalize(seq):
        return sorted(set(seq))

    for seed in seeds:
        B = normalize(seed)
        # ensure 0 in B
        if 0 not in B:
            B = normalize(B + [0])

        # keep a compact limit on size
        max_iterations = 100
        current_score = score_of(B)

        if current_score > best_score:
            best_overall = list(B)
            best_score = current_score

        # hill-climb: try inserting new element in a restricted range
        for it in range(max_iterations):
            improved = False
            candidates = []
            for x in range(0, max_elem_bound + 1):
                if x in B:
                    continue
                trial = normalize(B + [x])
                n = compute_n(trial)
                if n <= 0:
                    continue  # must cover 1
                s = - (len(trial) ** 2) / n
                candidates.append((s, trial, n))
            if not candidates:
                break

            candidates.sort(reverse=True, key=lambda t: t[0])
            s, trial, n = candidates[0]
            if s <= current_score:
                break  # no improvement
            B = list(trial)
            current_score = s
            improved = True

            if current_score > best_score:
                best_overall = list(B)
                best_score = current_score

            if len(B) > 60:  # safety: don't explode the set
                break

            if not improved:
                break

        # end seed loop

    # Final cleanup: ensure 0 is present and deduplicate
    if 0 not in best_overall:
        best_overall.append(0)
    best_overall = sorted(set(best_overall))
    return best_overall
