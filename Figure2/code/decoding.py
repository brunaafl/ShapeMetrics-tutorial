"""Decoding brain-region identity from single-neuron encoding profiles.

Posani, Wang et al. (2026) ask whether a neuron's response profile says where the
neuron is. Their released multi-area script answers it with clustering plus a Rand
index; here the same question is put to a *decoder*, one pair of regions at a time:
given the encoding profile of a single neuron, which of the two regions is it from?

Everything is per PAIR, which is what makes the result relatable to the anatomy --
a pair has a distance along the cortical hierarchy, an anatomical connectivity and a
Procrustes shape distance, and the decoding accuracy can be regressed on all of them.

The two feature spaces are the ones `data.Dataset.features` already defines:
"selectivity" (8 variables, the space of their Fig. 3e-f) and "temporal" (the full
8 x 100 coefficients).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_SUB, N_REPEATS, N_FOLDS = 50, 20, 5    # as in `shape.distance_matrix`


def classifier(C=1.0):
    """Standardise, then regularised logistic regression.

    Standardisation is not cosmetic: the eight selectivity coefficients differ in
    scale by an order of magnitude, and an unscaled penalty would fall almost
    entirely on the small ones.
    """
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, max_iter=5000))


def pair_accuracy(Xa, Xb, n_folds=N_FOLDS, seed=0, shuffle=False, model=None):
    """Cross-validated accuracy at telling a neuron of Xa from a neuron of Xb.

    The two groups have the same number of neurons, so chance is 0.5 exactly and
    accuracy needs no balancing. `shuffle` permutes the labels: the same estimator
    and the same folds, with nothing left to learn.
    """
    X = np.vstack([Xa, Xb])
    y = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
    if shuffle:
        y = np.random.default_rng(seed).permutation(y)
    cv = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    return cross_val_score(model or classifier(), X, y, cv=cv).mean()


def _samples(data, F, rng, n_sub):
    """One equal-sized draw of neurons from every region."""
    return [F[rng.choice(data.neurons(a), n_sub, replace=False)] for a in data.regions]


def decoding_matrix(data, mode="selectivity", n_sub=N_SUB, n_repeats=N_REPEATS,
                    n_folds=N_FOLDS, seed=0, shuffle=False, model=None, progress=None):
    """K x K matrix of pairwise region-decoding accuracies.

    Every region is subsampled to `n_sub` neurons before each pair is decoded, so
    that a region's size can drive neither the class balance nor the amount of
    training data, and the matrix is averaged over `n_repeats` draws -- the same
    protection `shape.distance_matrix` applies to the shape distances.
    """
    rng = np.random.default_rng(seed)
    F = data.features(mode)
    K = len(data.regions)
    A = np.zeros((K, K))
    it = range(n_repeats)
    it = progress(it) if progress else it
    # one seed per repeat rather than per pair: within a repeat every pair then
    # gets the same partition of the indices 0..2*n_sub-1 into folds, but the
    # neurons behind those indices are drawn afresh for each pair, so nothing is
    # shared between pairs except an arbitrary numbering
    for r in it:
        S = _samples(data, F, rng, n_sub)
        for i in range(K):
            for j in range(i + 1, K):
                A[i, j] += pair_accuracy(S[i], S[j], n_folds, seed=seed + r,
                                         shuffle=shuffle, model=model)
    A /= n_repeats
    return A + A.T


def split_half(data, mode="selectivity", n_sub=N_SUB, n_repeats=N_REPEATS,
               n_folds=N_FOLDS, seed=0, model=None):
    """Decoding an arbitrary split of one region against itself: the noise floor.

    A between-region accuracy only means something above this. Unlike a label
    shuffle it keeps the labels attached to the neurons, so any accuracy it finds
    is real structure inside the region rather than an artefact of the split.
    """
    rng = np.random.default_rng(seed)
    F = data.features(mode)
    out = np.zeros(len(data.regions))
    for _ in range(n_repeats):
        for k, a in enumerate(data.regions):
            h = rng.choice(data.neurons(a), 2 * (n_sub // 2), replace=False)
            out[k] += pair_accuracy(F[h[:n_sub // 2]], F[h[n_sub // 2:]],
                                    n_folds, seed=seed, model=model)
    return out / n_repeats


def decodability(A):
    """How distinguishable each region is from all the others: its mean row."""
    K = len(A)
    return (A.sum(1)) / (K - 1)          # the diagonal is zero and drops out
