"""The clustering test: sweep k, keep the best silhouette, compare with a null.

Silhouette cannot score the one-cluster hypothesis -- it needs at least two
clusters -- so "no clusters" is simulated rather than evaluated: draws from a
single Gaussian matched to the data's mean and covariance, swept identically.
The null makes the same free choice of k, so choosing k by argmax costs nothing.

The statistic is Posani, Wang, Muscinelli, Paninski & Fusi (2026)'s; their
pipeline is imported from the vendored clone rather than reimplemented, via the
thin wrappers at the bottom of this module.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def _posani():
    """The Posani et al. (2026) pipeline, vendored under vendor/posani_utils.

    It used to be reached by sys.path-inserting Posani/clustering-analysis, a
    git clone inside an analysis directory -- so the shared library depended on
    an analysis, and neither could move. Now it is a normal subpackage; see
    vendor/README.md for provenance.
    """
    from .vendor.posani_utils.clustering_algo import clustering   # noqa: E402
    from .vendor.posani_utils.dr_algo import dr                   # noqa: E402
    return clustering, dr


# ------------------------------------------------------------------ the sweep
def silhouette_sweep(E, ks, seed=42, n_init=50):
    """Best-of-`n_init` k-means silhouette at each k, for a set of points."""
    return np.array([silhouette_score(E, KMeans(n_clusters=k, n_init=n_init,
                     init="random", random_state=seed).fit_predict(E)) for k in ks])


def best_silhouette(E, ks, seed=42, n_init=50):
    """The statistic: sweep k, keep the best."""
    return silhouette_sweep(E, ks, seed=seed, n_init=n_init).max()


def best_silhouette_and_k(E, ks, seed=42, n_init=50):
    """`best_silhouette`, plus the number of clusters that won."""
    sweep = silhouette_sweep(E, ks, seed=seed, n_init=n_init)
    return float(sweep.max()), int(ks[sweep.argmax()])


# ------------------------------------------------------------------ the nulls
def gaussian_null(E, ks, n_draw=100, seed=0, progress=None):
    """The same sweep on draws from ONE Gaussian matched to E's mean and
    covariance -- the real spread, none of the lumpiness.

    Returns the full `n_draw x len(ks)` array of sweeps; take `.max(1)` to let
    each draw choose its own best k, as the data does.
    """
    rng = np.random.default_rng(seed)
    mu, S = E.mean(0), np.cov(E.T)
    it = range(n_draw) if progress is None else progress(n_draw)
    return np.array([silhouette_sweep(rng.multivariate_normal(mu, S, len(E)), ks)
                     for _ in it])


def curve_gaussian_null(X, rng):
    """One Gaussian draw with X's mean and empirical covariance, in the ORIGINAL
    feature space rather than an embedding.

    For tests whose pipeline is nonlinear (standardise, PCA, cluster): drawing
    here and pushing the draw through the identical pipeline makes that
    nonlinearity hit data and null alike.
    """
    w, V = np.linalg.eigh(np.cov(X - X.mean(0), rowvar=False))
    return X.mean(0) + rng.standard_normal((len(X), len(w))) \
        * np.sqrt(np.clip(w, 0, None)) @ V.T


def null_stats(obs, null):
    """z and the one-sided permutation p of an observed value against a null."""
    null = np.asarray(null)
    return dict(obs=float(obs), null=null,
                z=float((obs - null.mean()) / (null.std() + 1e-12)),
                p=float((np.sum(null >= obs) + 1) / (len(null) + 1)))


# ------------------------------------------------- the vendored Posani pipeline
def clustering_space(X, n_comp=None, scale=True):
    """Standardise each neuron across its features, then PCA.

    `scale=False` centres each neuron without dividing by its standard deviation.
    The division is what Posani et al.'s pipeline does, and it is right when every
    neuron carries real tuning: it stops loud neurons from dominating. It is wrong
    when a large fraction of the units are unreliable, because dividing a
    near-flat curve by its own tiny standard deviation turns noise into a
    full-amplitude curve, and a cloud of those is close to uniform on the sphere
    -- which is maximally unclusterable and drags the whole sample below its own
    null. Centring keeps amplitude as the signal it is.

    By default PCA keeps 90% of the variance, which is Posani et al.'s rule and
    lets the data choose its own dimensionality. Pass `n_comp` to fix it instead.

    Fixing it matters when data and null are compared: silhouette falls as
    dimensionality rises, so if the null happens to land in fewer dimensions than
    the data it wins on dimensionality rather than on lumpiness. The 90% rule is
    safe when the two land in the same place and misleading when they do not --
    which is worth checking rather than assuming.
    """
    _, dr = _posani()
    Xs = X - X.mean(1, keepdims=True)
    if scale:
        Xs = Xs / X.std(1, keepdims=True)
    kw = dict(method="pca", exp_var=0.9) if n_comp is None else \
        dict(method="pca", ncomp=int(n_comp))
    return dr(Xs, kw)


def pipeline_silhouette(X, k_lim=(2, 11), n_init=10, n_comp=None, scale=True):
    """Best mean silhouette over k, for neurons put through `clustering_space`."""
    clustering, _ = _posani()
    return float(clustering(clustering_space(X, n_comp, scale), "kmeans",
                            n_clus_lim=list(k_lim),
                            dis_metric="euclidean", n_init=n_init)["sscores_mean"])


def capped_silhouette(Z, n_init=50):
    """Best mean silhouette for an already-embedded subset, with k capped by
    sample size (k_max = n/10, at least 2, at most 10).

    The cap is what makes a real group and its size-matched pseudo-groups sweep
    an identical range of k, which they must for the comparison to be fair.
    """
    clustering, _ = _posani()
    kmax = max(2, min(10, len(Z) // 10))
    return clustering(Z, "kmeans", n_clus_lim=[2, kmax + 1],
                      dis_metric="euclidean", n_init=n_init)["sscores_mean"]


def condition_space_test(X, save_id, folder, n_null=100, k_lim=(3, 21),
                         n_init=50, min_n=50):
    """Posani et al.'s neuron-level clustering analysis, unmodified.

    Standardise each neuron across its features, PCA to 90% of the variance,
    sweep k, keep the best mean silhouette, and compare with `n_null` draws from
    a Gaussian matched to the data in the ORIGINAL feature space -- each draw
    preprocessed and swept identically.

    Returns obs / null / z / labels. The suspicious-cluster loop of their
    pipeline is off: it drops clusters carried by a single session, which is a
    within-session artefact check that does not apply to the questions asked here.
    """
    from .vendor.posani_utils.clustering_analysis import (        # noqa: E402
        cluster_analysis_condition_space)

    r = cluster_analysis_condition_space(
        X,
        {"min_N": min_n, "algo": "kmeans", "n_init": n_init,
         "n_clus_lim": list(k_lim), "dis_metric": "euclidean", "save_id": save_id},
        {"remove_sus_clus": False, "sessions_orig": np.zeros(len(X), int),
         "sus_clus_thres": 0.9},
        {"N_null": n_null, "null_dist": "Gaussian"},
        dict(plot=False, folder=str(folder), save_id=save_id))
    return dict(obs=float(r["sscore_mean"]), null=np.asarray(r["sscore_nulls"]),
                z=float(r["sscore_z"]), labels=np.asarray(r["clus_labels"]))
