import numpy as np

def _score(points):
    # points: (16,2)
    # compute all pairwise distances
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    # ignore diagonal
    d = d + np.eye(d.shape[0]) * 1e6
    min_d = d.min()
    max_d = d.max()
    if min_d <= 0:
        return -np.inf
    ratio2 = (max_d / min_d) ** 2
    return -ratio2

def construct():
    """
    Initialize from a jittered 4x4 grid in [0,1]^2 and apply simulated-annealing style stochastic optimization
    to reduce the ratio max/min pairwise distance. Returns a (16,2) array.
    """
    # Start from a 4x4 grid with small jitter to avoid perfect degeneracy
    g = np.linspace(0.0, 1.0, 4)
    grid = np.array([[x, y] for x in g for y in g], dtype=float)
    rng = np.random.default_rng(seed=12345)
    init_jitter = 0.04
    pts = grid + rng.normal(scale=init_jitter, size=grid.shape)

    # Bounds to keep search in a reasonable region while allowing exploration
    lower, upper = -0.2, 1.2

    best_pts = pts.copy()
    best_score = _score(best_pts)

    # SA-like parameters
    iterations = 6000
    t0 = 0.15  # initial "temperature" controlling step size
    tf = 0.001  # final temperature
    for it in range(iterations):
        # temperature schedule (exponential decay)
        t = t0 * (tf / t0) ** (it / max(1, iterations - 1))

        # propose: perturb a random point
        i = rng.integers(0, best_pts.shape[0])
        candidate = best_pts.copy()
        # move magnitude scales with temperature
        scale = 0.1 * t * (1.0 + rng.random() * 0.5)
        candidate[i] += rng.normal(scale=scale, size=2)

        # clamp to bounds to keep finite and in a reasonable region
        candidate[i] = np.clip(candidate[i], lower, upper)

        sc = _score(candidate)
        # Accept if improved or with a small probability if worse
        if sc > best_score or rng.random() < np.exp((sc - best_score) / max(1e-9, t)):
            best_pts = candidate
            best_score = sc

        # Optional occasional local reheat around the current best to escape shallow minima
        if (it % 1500) == 0:
            jitter = rng.normal(scale=0.05, size=best_pts.shape)
            alt = best_pts + jitter
            alt = np.clip(alt, lower, upper)
            sc_alt = _score(alt)
            if sc_alt > best_score:
                best_pts = alt
                best_score = sc_alt

    # Final return: ensure finite coordinates; scaling is free, ratio invariant
    return best_pts.astype(float)
