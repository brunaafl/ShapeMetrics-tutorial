"""Distances between neural populations, run over many pairs at once.

`dsd` and `ssd` are the deterministic and stochastic shape distances used by the
head-direction, fMRI and IBL analyses. They were defined in
`ibl_analyses/scripts/utils.py`, which is not tracked by this repository -- the
whole `ibl_analyses/` tree is local-only -- so notebooks that imported them
could not run from a clone. They live here now; the original file is untouched
for the local IBL pipeline that still uses it.

Ray is imported lazily, so importing `shapemetrics` costs nothing if these are
not used.
"""
import numpy as np
from netrep.metrics import GaussianStochasticMetric, LinearMetric


def _pair_workers():
    """The two per-pair distance functions, as ray remote tasks."""
    import ray

    @ray.remote
    def stochastic(pair, alpha=2., niter=1000):
        Xi, Xj = pair
        metric = GaussianStochasticMetric(alpha, niter=niter)
        metric.fit(Xi, Xj)
        return metric.score(Xi, Xj)

    @ray.remote
    def deterministic(pair, alpha=0.):
        Xi, Xj = pair
        metric = LinearMetric(alpha=alpha, center_columns=True,
                              score_method="euclidean")
        metric.fit(Xi, Xj)
        return metric.score(Xi, Xj)

    return ray, stochastic, deterministic


def ssd(pairs, alpha=2., niter=1000):
    """Stochastic shape distance for every (Xi, Xj) in `pairs`."""
    ray, stochastic, _ = _pair_workers()
    return np.array(ray.get([stochastic.remote(p, alpha, niter) for p in pairs]))


def dsd(pairs, alpha=0.):
    """Deterministic shape distance for every (Xi, Xj) in `pairs`.

    alpha = 0 is CCA-like (whitened), alpha = 1 is Procrustes.
    """
    ray, _, deterministic = _pair_workers()
    return np.array(ray.get([deterministic.remote(p, alpha) for p in pairs]))


def pair_distance(A, B, alpha=0.):
    """The same distance for a single pair, in-process -- no ray, no cluster."""
    metric = LinearMetric(alpha=alpha, center_columns=True,
                          score_method="euclidean")
    metric.fit(A, B)
    return metric.score(A, B)
