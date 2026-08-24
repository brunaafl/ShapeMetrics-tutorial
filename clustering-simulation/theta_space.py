"""The region_space simulation, with the clustered variables made CONDITIONS.

Everything here is `region_space` with one thing changed: what the two axes of the
neuron cloud are.

  region_space : a neuron is a tuning SHAPE -- (skew, width) -- and its tuning curve
                 is a LINEAR function of those two numbers.  The population geometry
                 is then quadratic in them, so it depends only on the cloud's mean and
                 covariance, and `a` (which is covariance-preserving) is invisible to it.

  theta_space  : a neuron is a conjunctive 2-D von Mises cell with preferred angles
                 (mu, nu) for two task variables theta1, theta2.  Every neuron has the
                 SAME width and skew; only the preferred angles differ.  The angles
                 enter the tuning NONLINEARLY, so the geometry depends on all Fourier
                 moments of the (mu, nu) distribution rather than just the first two --
                 and clustering can therefore leak into it.

`a` and `psi` mean exactly what they did before, now in the (mu, nu) plane: `a` sets how
bimodal the cloud of preferred angles is along its long axis, `psi` the orientation of
that axis.

Every analysis function is imported from `region_space` unmodified, so the two notebooks
differ only in the generator.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import region_space as _R                                            # noqa: E402
from region_space import (  # noqa: F401,E402  -- the analyses, unchanged
    AMOUNT_MAX, AMOUNT_MIN, EMB_DIM, KLIM, NINIT, N_KINDS, N_NEURONS, N_PCS,
    N_REGIONS, PSI, PSI_SPAN, SIG_L, SIG_S, best_silhouette, categoricality,
    continuum_or_types, mean_embed, pca_embed, procrustes_distances)

# ------------------------------------------------------- the conditions
NG = 16                        # grid per variable -> NG*NG conditions
KAPPA = 2.0                    # tuning width, identical for every neuron
ANG = 1.2                      # radians spanned by one unit of the weight cloud
NOISE = _R.NOISE

_g = np.linspace(-np.pi, np.pi, NG, endpoint=False)
TH1, TH2 = (a.ravel() for a in np.meshgrid(_g, _g, indexing="ij"))


def _wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def region_angles(amount, psi, rng, n_neurons=N_NEURONS):
    """Preferred angles (mu, nu) of one region's neurons.

    Identical construction to `region_space.region_weights` -- a cloud elongated
    along `psi`, bimodal along that axis by `amount`, with the variance along each
    axis fixed -- but the two coordinates are now preferred ANGLES rather than
    tuning-shape coefficients.
    """
    return _wrap(ANG * _R.region_weights(amount, psi, rng, n_neurons))


def simulate(amounts, psis, seed=0, n_neurons=N_NEURONS):
    """Conjunctive 2-D von Mises tuning over the theta1 x theta2 grid."""
    rng = np.random.default_rng(seed)
    A = np.concatenate([region_angles(a, p, rng, n_neurons)
                        for a, p in zip(amounts, psis)])
    vm = lambda x: np.exp(KAPPA * (np.cos(x) - 1.0))                  # noqa: E731
    X = vm(TH1[None] - A[:, :1]) * vm(TH2[None] - A[:, 1:])           # neurons x conds
    return (X + NOISE * rng.standard_normal(X.shape),
            np.repeat(np.arange(len(amounts)), n_neurons))


def scenario(name, seed=0, n_regions=N_REGIONS):
    """Byte-for-byte the logic of `region_space.scenario`, on the new generator."""
    if name == "continuum":
        t = np.sort(np.random.default_rng(seed).uniform(0, 1, n_regions))
        amounts = AMOUNT_MIN + (AMOUNT_MAX - AMOUNT_MIN) * t
        psis = (-PSI + 2 * PSI * t) * PSI_SPAN
        colour = t
    elif name == "no_gradient":
        rng = np.random.default_rng(seed)
        t = np.sort(rng.uniform(0, 1, n_regions))
        amounts = rng.permutation(AMOUNT_MIN + (AMOUNT_MAX - AMOUNT_MIN) * t)
        psis = (-PSI + 2 * PSI * t) * PSI_SPAN
        colour = t
    elif name == "kinds":
        # `a` = 0 throughout: no region is categorical, all the structure is in
        # `kind`.  Mirrors `region_space.scenario`, which this function is meant
        # to match exactly so that the GENERATOR is the only difference between
        # the two modules.
        amounts = np.zeros(n_regions)
        psis = np.tile(np.linspace(-PSI, PSI, N_KINDS), n_regions // N_KINDS)
        colour = np.tile(np.arange(N_KINDS), n_regions // N_KINDS)
    else:
        raise ValueError(name)
    X, region = simulate(amounts, psis, seed=seed, n_neurons=N_NEURONS)
    return X, region, colour


def preferred_angles(X):
    """Each neuron's preferred (theta1, theta2), measured from its responses --
    the analogue of projecting onto the shape modes in `region_space`."""
    R2 = X.reshape(len(X), NG, NG)
    out = []
    for M in (R2.mean(2), R2.mean(1)):
        M = np.clip(M - M.min(1, keepdims=True), 0, None)
        z = (M * np.exp(1j * _g)[None]).sum(1) / (M.sum(1) + 1e-12)
        out.append(np.angle(z))
    return np.c_[out[0], out[1]]
