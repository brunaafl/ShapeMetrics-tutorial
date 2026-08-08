"""Simulate a cohort of individuals, each with a neural geometry AND a behavior.

Each individual encodes a shared circular variable theta with band-limited
periodic tuning.  An individual's *harmonic weight profile* sets the shape of
its neural manifold, and also sets its behavioral acuity -- so behavior is
readable from geometry, which is the premise the shape-metrics framework rests
on.

Three ground-truth TYPES correspond to three harmonic profiles (broad -> sharp
tuning).  Independently of type, individuals differ in how many *other*
variables they encode ("richness").  Richness is irrelevant to behavior, but it
is exactly what linear-regression predictivity is most sensitive to: a richer
individual can linearly predict a simpler one, never the reverse.

Run:  python simulate_data.py
Out:  simulated_data.npz
"""
import numpy as np

# ---------------------------------------------------------------- constants
N_NEURONS = 60
N_THETAS = 200
N_INDIVIDUALS = 30
N_HARMONICS = 4
N_NUISANCE_VARS = 4          # size of the shared pool of "other" variables
RICHNESS_WEIGHT = 0.10       # see the sweep in analyses.py: at this level the
                             # nuisance is invisible to Procrustes (r = 0.07)
                             # but still dominates predictivity.  Above ~0.2 it
                             # starts to contaminate Procrustes too.
NOISE = 0.02

THETAS = np.linspace(-np.pi, np.pi, N_THETAS, endpoint=False)

# Harmonic weight profiles.  Every individual spans the SAME 2*N_HARMONICS
# dimensional function space, so individuals differ by an invertible linear
# reweighting -- precisely the transformation linear regression is blind to.
PROFILES = {
    0: np.array([1.00, 0.15, 0.05, 0.02]),   # low harmonics  -> broad tuning
    1: np.array([1.00, 0.70, 0.35, 0.15]),   # mid
    2: np.array([1.00, 0.95, 0.85, 0.70]),   # high harmonics -> sharp tuning
}


# ---------------------------------------------------------------- generators
def tuning_curves(weights, rs):
    """N_NEURONS x N_THETAS band-limited periodic tuning.

    Neuron n has preferred direction mu_n and response
        f_n(theta) = sum_h w_h cos(h (theta - mu_n)).
    """
    mu = rs.uniform(-np.pi, np.pi, N_NEURONS)
    X = sum(
        weights[h - 1] * np.cos(h * (THETAS[None, :] - mu[:, None]))
        for h in range(1, N_HARMONICS + 1)
    )
    return X / np.abs(X).max()


def nuisance_block(nuisance_thetas, n_encoded, rs):
    """Response to `n_encoded` of the shared nuisance variables."""
    Z = np.zeros((N_NEURONS, N_THETAS))
    for c in range(n_encoded):
        mu = rs.uniform(-np.pi, np.pi, N_NEURONS)
        block = np.cos(nuisance_thetas[None, :, c] - mu[:, None])
        Z += block / np.abs(block).max()
    return Z


def acuity(weights):
    """Fisher-information-like discrimination acuity from the harmonic profile.

    Depends on every harmonic weight, so predicting it is not a disguised 1-D
    regression on a single generative scalar.
    """
    h = np.arange(1, N_HARMONICS + 1)
    return np.log(np.sum((h ** 2) * weights ** 2))


def simulate(seed=0, richness_weight=RICHNESS_WEIGHT, noise=NOISE):
    """Return (Xs, behavior, labels, weights, richness).

    Xs : list of N_NEURONS x N_THETAS arrays, one per individual
    """
    rs = np.random.RandomState(seed)

    labels = np.repeat([0, 1, 2], N_INDIVIDUALS // 3)
    weights = np.array([PROFILES[t] * rs.uniform(0.9, 1.1, N_HARMONICS)
                        for t in labels])
    richness = rs.randint(0, N_NUISANCE_VARS + 1, size=N_INDIVIDUALS)

    # Nuisance variables are SHARED across individuals, like theta.
    nuisance_thetas = rs.uniform(-np.pi, np.pi, size=(N_THETAS, N_NUISANCE_VARS))

    Xs = []
    for w, r in zip(weights, richness):
        X = tuning_curves(w, rs)
        X = X + richness_weight * nuisance_block(nuisance_thetas, int(r), rs)
        X = X + noise * rs.standard_normal(X.shape)
        Xs.append(X - X.mean(axis=1, keepdims=True))

    behavior = np.array([acuity(w) for w in weights]) + rs.normal(0, 0.03, N_INDIVIDUALS)
    return np.array(Xs), behavior, labels, weights, richness


if __name__ == "__main__":
    Xs, behavior, labels, weights, richness = simulate(seed=0)

    ranks = [np.linalg.matrix_rank(X) for X in Xs]
    print(f"{N_INDIVIDUALS} individuals, {N_NEURONS} neurons x {N_THETAS} conditions")
    print(f"population rank: min {min(ranks)}, max {max(ranks)} "
          f"(expect >= {2 * N_HARMONICS} from the harmonic basis)")
    print(f"richness (# extra variables encoded): {np.bincount(richness)}")
    print(f"behavior range: {behavior.min():.2f} to {behavior.max():.2f}")

    np.savez(
        "simulated_data.npz",
        Xs=Xs, behavior=behavior, labels=labels, weights=weights,
        richness=richness, THETAS=THETAS,
        N_NEURONS=N_NEURONS, N_THETAS=N_THETAS, N_HARMONICS=N_HARMONICS,
        RICHNESS_WEIGHT=RICHNESS_WEIGHT,
    )
    print("wrote simulated_data.npz")
