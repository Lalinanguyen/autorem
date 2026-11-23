import time
import random

def compute_n_bitset(B, max_diff=4096):
    """
    Compute n using a bitset representation.
    B: iterable of non-negative ints
    max_diff: cap for differences to check (safety to stay within memory)
    Returns 0 if 1 is not a diff, else the largest n such that 1..n all diffs exist.
    """
    if not B:
        return 0
    B = sorted(set(B))
    if B and B[0] != 0:
        # ensure 0 is present as a baseline
        B = [0] + B
    bits = 0
    for v in B:
        bits |= (1 << v)
    # If 1 is not a diff with 0, then 1 is missing
    # A diff d exists iff there exists a and a+d in B; equivalently (bits & (bits >> d)) != 0
    if (bits & (bits >> 1)) == 0:
        return 0
    n = 0
    # Limit max_diff to avoid excessive loops; stop when next diff missing
    max_check = min(max_diff, max(B) if B else 0)
    while n + 1 <= max_check:
        d = n + 1
        if (bits & (bits >> d)) != 0:
            n += 1
        else:
            break
    return n

def construct():
    random.seed(1337)
    start_time = time.time()
    TIME_LIMIT = 1.8  # seconds

    # baseline: ensure 0 and 1 present
    base = [0, 1]
    best_B = sorted(set(base))
    best_n = compute_n_bitset(best_B)
    best_score = - (len(best_B) ** 2) / max(1, best_n)

    MAX_B = 2000  # dedupe cap
    MAX_VALUE = 2000  # search space cap for candidate values

    def prune_after_add(Bcand, allow_removal_check=True):
        """
        Prune elements that are redundant with respect to n.
        If removing an element does not decrease n, drop it.
        """
        Bcur = sorted(set(Bcand))
        if 0 not in Bcur:
            Bcur = [0] + Bcur
        current_n = compute_n_bitset(Bcur)
        improved = True
        while improved:
            improved = False
            # try removing each element (except 0)
            for x in list(Bcur):
                if x == 0:
                    continue
                nb = [y for y in Bcur if y != x]
                if not nb:
                    continue
                nn = compute_n_bitset(nb)
                if nn >= current_n:
                    Bcur = sorted(set(nb))
                    current_n = nn
                    improved = True
                    break
        return Bcur

    def candidate_by_b_plus(n, Bcur):
        """
        Generate biased candidates: b + (n+1) for b in Bcur, within range.
        """
        cands = []
        step = n + 1
        for b in Bcur:
            v = int(b + step)
            if 0 <= v <= MAX_VALUE:
                cands.append(v)
        # also include some random tweaks to escape local optima
        for _ in range(2):
            cands.append(random.randint(0, MAX_VALUE))
        return list(set(cands))

    # initial candidate pools
    pool = []
    pool.append(sorted(set(base)))
    # a few randomized seeds to explore
    for _ in range(4):
        seeds = {0, 1}
        for _t in range(random.randint(0, 2)):
            seeds.add(random.randint(0, MAX_VALUE // 2))
        pool.append(sorted(seeds))

    # ensure best found so far included
    pool.append(best_B)

    # main loop
    while time.time() - start_time < TIME_LIMIT:
        Bcand = random.choice(pool)
        Bcand = sorted(set(Bcand))
        if len(Bcand) == 0:
            Bcand = [0]

        # evaluate current
        n = compute_n_bitset(Bcand)
        if n > 0:
            score = - (len(Bcand) ** 2) / n
            if score > best_score:
                best_score = score
                best_B = prune_after_add(Bcand)
                best_n = compute_n_bitset(best_B)
                # update pool with pruned better candidate
                pool.append(sorted(set(best_B)))
        
        # extend with biased candidates
        Bi = prune_after_add(Bcand)
        candid_ext = candidate_by_b_plus(n if n>0 else 1, Bi)
        for c in candid_ext:
            nb = sorted(set(Bi + [c]))
            if len(nb) > MAX_B:
                continue
            nb = prune_after_add(nb)
            if nb and nb not in pool:
                pool.append(nb)

        # occasional random prune of pool to keep search light
        if len(pool) > 60:
            random.shuffle(pool)
            pool = pool[:60]

    final_B = sorted(set(best_B))
    if 0 not in final_B:
        final_B = [0] + final_B
    if len(final_B) > 2000:
        final_B = final_B[:2000]
    return final_B
