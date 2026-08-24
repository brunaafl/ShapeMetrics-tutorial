"""
Procrustes shape distance between two neural populations.

Same quantity as netrep's LinearMetric(alpha=1.0, score_method="euclidean"):
both matrices are column-centered, then

    d(X, Y) = min_{Q orthogonal} || X - Y Q ||_F

which has the closed form  sqrt( ||X||^2 + ||Y||^2 - 2 * sum(svdvals(X^T Y)) ).
X and Y must share the same number of rows (conditions) but may have been
reduced to the same number of columns (neurons / PCs) beforehand.
"""
import numpy as np
from sklearn.decomposition import PCA


def center(X):
    return X - X.mean(axis=0, keepdims=True)


def procrustes_distance(X, Y):
    X, Y = center(X), center(Y)
    s = np.linalg.svd(X.T @ Y, compute_uv=False).sum()
    d2 = (X ** 2).sum() + (Y ** 2).sum() - 2 * s
    return np.sqrt(max(d2, 0.0))


def preprocess(X, n_pcs):
    """Conditions x neurons -> conditions x n_pcs, unit Frobenius norm."""
    X = PCA(n_pcs).fit_transform(X)      # also centers the columns
    return X / np.linalg.norm(X)


def pairwise(Xs):
    K = len(Xs)
    D = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            D[i, j] = D[j, i] = procrustes_distance(Xs[i], Xs[j])
    return D
