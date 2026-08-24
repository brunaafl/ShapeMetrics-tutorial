"""Putting a distance matrix into coordinates, because k-means needs them."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS


def mds(D, n_components=5, seed=0, n_init=4):
    """SMACOF MDS on a precomputed dissimilarity matrix."""
    return MDS(n_components=n_components, dissimilarity="precomputed",
               random_state=seed, normalized_stress=False,
               n_init=n_init).fit_transform(D)


def classical_mds(D, n_components=5):
    """Classical MDS (PCoA): eigendecomposition of the doubly-centred -D^2/2.

    Deterministic, and its axes come out ordered by variance -- unlike SMACOF,
    whose components are in arbitrary order. Use this when the coordinates are
    read one by one (as regressors, or as "PC1 recovers the gradient").
    """
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    w, V = np.linalg.eigh(-0.5 * J @ (D ** 2) @ J)
    idx = np.argsort(w)[::-1][:n_components]
    return V[:, idx] * np.sqrt(np.clip(w[idx], 0, None))
