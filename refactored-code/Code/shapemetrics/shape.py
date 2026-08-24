"""Procrustes distance between neural populations.

Two flavours are in use in this project and both are kept, because they are not
the same number:

  `pair_distance`      netrep's LinearMetric on each population's PCs, no
                       normalisation -- the distance of the head-direction and
                       IBL subject-space analyses.
  `procrustes_distance` the closed form, on populations put through `preprocess`
                       (PCA + unit Frobenius norm) -- the distance of the Posani
                       region analyses, where regions are subsampled to equal
                       size and the scale has to be taken out.
"""
import numpy as np
from netrep.metrics import LinearMetric
from sklearn.decomposition import PCA


# --------------------------------------------------------------- netrep flavour
def pair_distance(A, B, alpha=1.0):
    """Shape distance between two `conditions x components` matrices."""
    m = LinearMetric(alpha=alpha, center_columns=True, score_method="euclidean")
    m.fit(A, B)
    return m.score(A, B)


def distance_matrix(X, labels, n_pcs=20, alpha=1.0):
    """Each group of `labels` becomes one point cloud; all pairs compared.

    X is `neurons x features`; a group's population is transposed to
    `features x neurons` and reduced to `n_pcs` components, so every group ends
    up in the same `features x n_pcs` shape whatever its neuron count.
    """
    # explicit solver: "auto" switches to the stochastic randomized SVD for
    # some shapes, which made distances depend on the global RNG state
    pca = PCA(n_components=n_pcs, svd_solver="full")
    P = [pca.fit_transform(X[labels == g].T) for g in np.unique(labels)]
    n = len(P)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = pair_distance(P[i], P[j], alpha)
    return D


# ----------------------------------------------------------- closed-form flavour
def center(X):
    return X - X.mean(axis=0, keepdims=True)


def procrustes_distance(X, Y):
    """d(X, Y) = min over orthogonal Q of || X - Y Q ||_F.

    Same quantity as LinearMetric(alpha=1, score_method="euclidean"), via
    sqrt( ||X||^2 + ||Y||^2 - 2 sum(svdvals(X'Y)) ).
    """
    X, Y = center(X), center(Y)
    s = np.linalg.svd(X.T @ Y, compute_uv=False).sum()
    return np.sqrt(max((X ** 2).sum() + (Y ** 2).sum() - 2 * s, 0.0))


def preprocess(X, n_pcs):
    """Conditions x neurons -> conditions x n_pcs, unit Frobenius norm."""
    X = PCA(n_pcs).fit_transform(X)      # also centers the columns
    return X / np.linalg.norm(X)


def pairwise(Xs):
    """Full distance matrix over a list of preprocessed populations."""
    n = len(Xs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = procrustes_distance(Xs[i], Xs[j])
    return D
