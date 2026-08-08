"""Shape distances between brain regions, and what can be regressed out of them."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

from procrustes import preprocess, pairwise, procrustes_distance   # noqa: F401


def distance_matrix(data, mode="temporal", n_pcs=25, n_sub=50, n_repeats=20, seed=0):
    """K x K matrix of Procrustes distances between the regions of `data`.

    Every region is subsampled to `n_sub` neurons so that region size cannot drive
    the distances, and the matrix is averaged over `n_repeats` subsamples.
    """
    rng = np.random.default_rng(seed)
    features = data.features(mode)
    idx = {a: data.neurons(a) for a in data.regions}
    D = np.zeros((len(data.regions), len(data.regions)))
    for _ in range(n_repeats):
        Xs = [preprocess(features[rng.choice(idx[a], n_sub, replace=False)].T, n_pcs)
              for a in data.regions]
        D += pairwise(Xs)
    return D / n_repeats


def split_half(data, mode="temporal", n_pcs=25, n_sub=50, n_repeats=20, seed=0):
    """Distance between two disjoint halves of each region: the noise floor."""
    rng = np.random.default_rng(seed)
    features = data.features(mode)
    out = np.zeros(len(data.regions))
    for _ in range(n_repeats):
        for k, a in enumerate(data.regions):
            h = rng.choice(data.neurons(a), 2 * (n_sub // 2), replace=False)
            out[k] += procrustes_distance(
                preprocess(features[h[:n_sub // 2]].T, n_pcs),
                preprocess(features[h[n_sub // 2:]].T, n_pcs))
    return out / n_repeats


def mds(D, n_components=10, seed=0):
    """Embed the dissimilarity matrix in Euclidean space, minimising distortion.

    The MDS solution is only defined up to a rotation, so we rotate it onto its
    principal axes. Distances are unchanged, but the individual coordinates become
    reproducible -- which matters as soon as they are used one by one, e.g. as
    regressors.
    """
    m = MDS(n_components=n_components, dissimilarity="precomputed",
            normalized_stress=False, random_state=seed, n_init=10).fit_transform(D)
    return PCA(n_components).fit_transform(m)


def classical_mds(D, n_components=10):
    """Classical MDS (PCoA): eigendecomposition of the doubly-centred -D^2/2.

    Unlike the SMACOF algorithm behind sklearn's MDS, this has a closed form -- no
    random restarts, no local minima, and axes fixed up to sign. Use it whenever the
    individual coordinates matter (as regressors, say) rather than just the distances:
    a SMACOF embedding of the same matrix gives different coordinates every run.
    """
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:n_components]
    return V[:, idx] * np.sqrt(np.clip(w[idx], 0, None))


def embed(D, n_components=10, seed=0, method="classical"):
    """The first two coordinates, for plotting, with their share of the variance."""
    e = (classical_mds(D, n_components) if method == "classical"
         else mds(D, n_components, seed))
    v = e.var(0)
    return e[:, :2], (v / v.sum())[:2]


def pairs(M):
    """Upper-triangular entries of a square matrix, as a flat vector."""
    return M[np.triu_indices(len(M), 1)]


def _design(X, y):
    X = np.atleast_2d(X)
    return X.T if X.shape[0] != len(y) else X


def score(y, pred):
    return r2_score(y, pred), spearmanr(y, pred)[0]


def _model(alphas=None, model=None):
    """Least squares, ridge, or any estimator factory passed in as `model`."""
    if model is not None:
        return model()
    return LinearRegression() if alphas is None else RidgeCV(alphas=alphas)


def fit(X, y, alphas=None, model=None):
    """The fitted model, when its weights or its in-sample predictions are wanted."""
    return _model(alphas, model).fit(_design(X, y), y)


def regress(X, y):
    """Plain least squares, fitted and scored on the same data (no cross-validation).

    Returns the predictions, R2 and the rank correlation with the target.
    """
    X = _design(X, y)
    pred = fit(X, y).predict(X)
    return pred, *score(y, pred)


def cv_regress(X, y, alphas=None, model=None):
    """Leave-one-out: every sample is predicted by a model fitted without it.

    With more predictors than samples can support, pass `alphas` to regularise (the
    penalty is then selected within each training fold, never on the held-out sample)
    or `model` for any other estimator factory, e.g. `lambda: LassoCV(cv=5)`.
    """
    X = _design(X, y)
    pred = np.empty(len(y), dtype=float)
    for k in range(len(y)):
        m = np.arange(len(y)) != k
        pred[k] = _model(alphas, model).fit(X[m], y[m]).predict(X[[k]])[0]
    return pred, *score(y, pred)


def cv_regress_truncated(X, y, dims=None):
    """Leave-one-out where the number of components is also chosen inside the fold.

    Keeping only the leading components is itself a form of regularisation; doing the
    choice honestly (an inner leave-one-out over the training regions) is what it
    costs when the right number is not known in advance.
    """
    X = _design(X, y)
    dims = np.arange(1, X.shape[1] + 1) if dims is None else np.asarray(dims)
    n_out = len(y)
    pred, chosen = np.empty(n_out), np.empty(n_out, dtype=int)
    for k in range(n_out):
        out = np.arange(n_out) != k
        Xtr, ytr = X[out], y[out]
        best = (-np.inf, dims[0])
        for n in dims:
            inner = np.empty(len(ytr))
            for j in range(len(ytr)):
                m = np.arange(len(ytr)) != j
                inner[j] = LinearRegression().fit(
                    Xtr[m, :n], ytr[m]).predict(Xtr[[j], :n])[0]
            best = max(best, (r2_score(ytr, inner), n))
        chosen[k] = best[1]
        pred[k] = LinearRegression().fit(Xtr[:, :chosen[k]], ytr).predict(
            X[[k], :chosen[k]])[0]
    return pred, *score(y, pred), chosen


def cv_regress_pairs(X, y, n_regions, alphas=None):
    """Leave-one-region-out for features defined on pairs of regions.

    Every pair that touches the held-out region goes to the test set, so no region
    contributes to both training and testing.
    """
    X = _design(X, y)
    ii, jj = np.triu_indices(n_regions, 1)
    pred = np.empty(len(y), dtype=float)
    for k in range(n_regions):
        test = (ii == k) | (jj == k)
        pred[test] = _model(alphas).fit(X[~test], y[~test]).predict(X[test])
    return pred, *score(y, pred)
