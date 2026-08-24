"""Region identity decoded from single neurons: the pairwise counterpart of `shape`.

`shape.distance_matrix` compares two groups as whole populations, up to an
orthogonal transform of their neurons. This asks the other question about the same
pair: given ONE neuron's feature vector, which of the two groups is it from?

The two are not the same test. A decoder needs only that the two clouds of feature
vectors have different distributions; a shape distance needs the populations to be
geometrically different after alignment. Either can move without the other, which is
what makes running both on one dataset informative.

The output is a K x K matrix of cross-validated accuracies. It is a dissimilarity in
the same sense the Procrustes matrix is -- 0.5 for two indistinguishable groups,
rising as they separate -- so it can be handed to the same embeddings and the same
clustering tests.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_FOLDS = 5


def classifier(C=1.0, max_iter=5000):
    """Standardise, then regularised logistic regression.

    Standardisation is not cosmetic: features on different scales would otherwise
    receive the same penalty, which puts it almost entirely on the small ones.
    """
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, max_iter=max_iter))


def pair_accuracy(A, B, n_folds=N_FOLDS, seed=0, shuffle=False, model=None):
    """Cross-validated accuracy at telling a row of A from a row of B.

    A and B are `neurons x features`. Give them the same number of rows and chance
    is 0.5 exactly, so the accuracy needs no balancing. `shuffle` permutes the
    labels: the same estimator and the same folds, with nothing left to learn.
    """
    X = np.vstack([A, B])
    y = np.r_[np.zeros(len(A)), np.ones(len(B))]
    if shuffle:
        y = np.random.default_rng(seed).permutation(y)
    cv = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    return float(cross_val_score(model or classifier(), X, y, cv=cv).mean())


def decoding_matrix(X, labels, n_folds=N_FOLDS, seed=0, shuffle=False, model=None,
                    n_sub=None, n_repeats=1, rng=None, progress=None):
    """K x K matrix of pairwise group-decoding accuracies.

    Signature mirrors `shape.distance_matrix`: X is `neurons x features` and the
    groups are the distinct values of `labels`, taken in sorted order.

    `n_sub` draws that many neurons from every group before each pair is decoded,
    and `n_repeats` averages over draws. Both are needed only when the groups differ
    in size -- otherwise a big group would be both easier to fit and unbalanced
    against a small one. With equal-sized groups the default (use them all, once) is
    already the right thing.
    """
    groups = np.unique(labels)
    K = len(groups)
    idx = [np.flatnonzero(labels == g) for g in groups]
    rng = np.random.default_rng(seed) if rng is None else rng
    A = np.zeros((K, K))
    it = range(n_repeats)
    it = progress(it) if progress else it
    for r in it:
        S = [X[i if n_sub is None else rng.choice(i, n_sub, replace=False)]
             for i in idx]
        for i in range(K):
            for j in range(i + 1, K):
                A[i, j] += pair_accuracy(S[i], S[j], n_folds, seed=seed + r,
                                         shuffle=shuffle, model=model)
    A /= n_repeats
    return A + A.T


def decodability(A):
    """How distinguishable each group is from all the others: its mean row."""
    return A.sum(1) / (len(A) - 1)      # the diagonal is zero and drops out
