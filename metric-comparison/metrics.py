"""The two dissimilarities being compared, plus metric-space diagnostics.

Procrustes  -- netrep LinearMetric(alpha=1), a proper metric.
Predictivity -- cross-validated ridge regression R^2, which is NOT symmetric.

Both operate on the same PCA-reduced data with the same number of components,
and the ridge penalty is chosen by inner cross-validation, so the comparison is
not stacked in favour of Procrustes.

NOTE: we deliberately do NOT import ibl_analyses/scripts/utils.py.  Its `dsd`
dispatches every single pair through a blocking ray round-trip, and its `alpha`
defaults to 0. (the whitened / CCA-like metric), not Procrustes.
"""
import itertools

import numpy as np
from netrep.metrics import LinearMetric
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

N_PCS = 10
RIDGE_ALPHAS = np.logspace(-4, 4, 17)


def to_pcs(Xs, n_pcs=N_PCS):
    """N_NEURONS x N_THETAS  ->  N_THETAS x n_pcs, identically for both metrics."""
    return [PCA(n_pcs).fit_transform(X.T) for X in Xs]


# ---------------------------------------------------------------- Procrustes
def d_procrustes(X, Y):
    """Procrustes size-and-shape distance.  Symmetric, satisfies the triangle
    inequality.  X, Y are (conditions x components)."""
    metric = LinearMetric(alpha=1, center_columns=True, score_method="euclidean")
    metric.fit(X, Y)
    return metric.score(X, Y)


def pairwise_procrustes(Ps):
    K = len(Ps)
    D = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            D[i, j] = D[j, i] = d_procrustes(Ps[i], Ps[j])
    return D


# ---------------------------------------------------------------- predictivity
def predictivity_r2(X, Y, n_splits=5, seed=0):
    """Cross-validated linear predictivity of Y from X, held out over conditions.

    This is the standard 'linear predictivity' / encoding-model score.  It is
    asymmetric: R2(X->Y) != R2(Y->X).
    """
    scores = []
    for train, test in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        model = RidgeCV(alphas=RIDGE_ALPHAS).fit(X[train], Y[train])
        scores.append(r2_score(Y[test], model.predict(X[test]),
                               multioutput="variance_weighted"))
    return float(np.mean(scores))


def pairwise_predictivity(Ps, seed=0):
    """Full asymmetric K x K dissimilarity, d(i->j) = 1 - R2(i->j)."""
    K = len(Ps)
    D = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i != j:
                D[i, j] = 1.0 - max(predictivity_r2(Ps[i], Ps[j], seed=seed), 0.0)
    return D


def symmetrize(D):
    """The obvious defence against asymmetry.  It repairs kNN but, as
    analyses.py shows, it does not restore the triangle inequality."""
    return (D + D.T) / 2


# ---------------------------------------------------------------- diagnostics
def asymmetry(D):
    """mean |D - D^T| / mean D.  Zero for any symmetric dissimilarity."""
    off = ~np.eye(len(D), dtype=bool)
    return float(np.abs(D - D.T)[off].mean() / D[off].mean())


def triangle_violations(D, tol=1e-12):
    """Fraction of ordered triples with d(a,b) > d(a,c) + d(c,b)."""
    K = len(D)
    bad = total = 0
    for i, j, k in itertools.combinations(range(K), 3):
        for a, b, c in ((i, j, k), (j, k, i), (k, i, j)):
            total += 1
            if D[a, b] > D[a, c] + D[c, b] + tol:
                bad += 1
    return bad / total


def negative_eigenvalue_mass(D):
    """Fraction of eigenvalue mass of the doubly-centred Gram matrix that is
    negative.  Zero iff D is Euclidean-embeddable; large values mean MDS and
    anything built on it are distorting the data."""
    K = len(D)
    J = np.eye(K) - np.ones((K, K)) / K
    G = -0.5 * J @ (D ** 2) @ J
    w = np.linalg.eigvalsh(G)
    return float(np.abs(w[w < 0]).sum() / np.abs(w).sum())
