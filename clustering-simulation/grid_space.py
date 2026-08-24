"""Within-region and across-region clustering as two independent knobs.

`region_space` contrasts three hand-picked fields.  This module makes the same
point as a 2-D design: the two kinds of clustering that were previously bundled
into named scenarios become two continuous axes, and each analysis is run over
the whole grid.

  a  WITHIN-region categoricality.  Every region's neurons are a cloud in
     skew-width space, bimodal along that region's own long axis by amount `a`.
     All regions in a cell share one `a`.  The construction holds the cloud's
     second moment fixed, so `a` provably does not move the region geometry.

  c  ACROSS-region clustering.  Region i is assigned to one of N_KINDS centres
     m_k spanning the psi range, and then

         psi_i = m_k + (1 - c) * u_i,   u_i ~ U(-delta/2, +delta/2)

     with delta the spacing between centres.  At c = 0 the psi values fill the
     range uniformly -- a continuum with no kinds in it; at c = 1 they sit
     exactly on N_KINDS values.  The marginal psi RANGE is the same at both
     ends, so `c` changes only how the regions are distributed within it and
     not how far apart the extremes are.  Without that, `c` would confound
     clustering with overall spread.

The four corners reproduce the earlier scenarios: (a=0, c=0) is structureless,
(a>0, c=0) is "gradient", (a=0, c=1) is "clustered", (a>0, c=1) is both.

TWO READOUTS, EACH AGAINST ITS OWN NULL.

  z_pooled   pool the neurons across regions and run the Posani pipeline
             (standardise each neuron, PCA to 90% variance, sweep k, best mean
             silhouette).  The null is Gaussian matched in RAW CURVE SPACE and
             pushed through the identical pipeline -- drawing it in PCA space
             instead inflates z badly.

  z_regions  compare whole region populations with Procrustes, embed with
             classical MDS, take the best silhouette over k.  The null is a
             Gaussian matched to the region embedding.

PSI_SCALE IS CALIBRATED, NOT CHOSEN.  The region-space test compares a
silhouette against a Gaussian, so it detects departure from a Gaussian rather
than clustering as such -- and a strongly curved continuum is such a departure.
Over the full psi range `region_space` returns p = 0.01 for a continuum with no
types in it.  PSI_SCALE therefore has to be picked so that the c = 0 column
really does sit at z ~ 0 while c = 1 still separates; see `calibrate`.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import region_space as R                                          # noqa: E402
import shapemetrics as sm                                         # noqa: E402

N_REGIONS = 60
N_NEURONS = 150                    # set by calibration: `c` = 1 separates only
                                   # when per-region sampling noise is small
N_KINDS = 3
PSI_SCALE = 0.55                   # set by calibrate(); see CALIBRATION below

# CALIBRATION OUTCOME.  There is NO psi range at which the c = 0 column sits at
# z = 0 while c = 1 separates.  Widening the range sharpens the lumps at c = 1
# but also curves the continuum at c = 0, and *reducing* the sampling noise
# sharpens both -- a cleaner continuum is a more clearly non-Gaussian arc.  At
# the chosen setting the c = 0 column sits at z ~ 2.5, which is the test's
# curvature floor rather than evidence of types.  The heatmap must therefore be
# read as a CONTRAST along c, not against z = 0.  This is a property of
# comparing a silhouette against a Gaussian, and it is why region_space.py
# sweeps only PSI_SPAN = 0.3 for its continuum scenario.

N_NULL = 25
N_POOL = 800                       # silhouette is O(n^2); the pooled cloud is
                                   # N_REGIONS * N_NEURONS neurons
KLIM, NINIT = R.KLIM, R.NINIT
KMAX_REGION, EMB_DIM, N_PCS = R.KMAX_REGION, R.EMB_DIM, R.N_PCS
KRANGE = range(2, KMAX_REGION + 1)


def field(a, c, seed=0, n_regions=N_REGIONS, psi_scale=PSI_SCALE,
          n_neurons=N_NEURONS):
    """One cell of the grid: `n_regions` regions at within-clustering `a` and
    across-clustering `c`."""
    rng = np.random.default_rng(seed)
    centres = np.linspace(-R.PSI, R.PSI, N_KINDS) * psi_scale
    delta = float(centres[1] - centres[0])
    kind = np.arange(n_regions) % N_KINDS
    psis = centres[kind] + (1 - c) * rng.uniform(-0.5, 0.5, n_regions) * delta
    amounts = np.full(n_regions, float(a))
    X, region = R.simulate(amounts, psis, seed=seed, n_neurons=n_neurons)
    return X, region, kind


def _z(obs, null):
    null = np.asarray(null, float)
    return float((obs - null.mean()) / (null.std() + 1e-12))


def z_pooled(X, rng, n_null=N_NULL, n_pool=N_POOL):
    """Neurons pooled across regions, clustered a la Posani."""
    idx = rng.choice(len(X), min(n_pool, len(X)), replace=False)
    A = X[idx]
    obs = sm.pipeline_silhouette(A, k_lim=KLIM, n_init=NINIT)
    null = [sm.pipeline_silhouette(sm.curve_gaussian_null(A, rng),
                                   k_lim=KLIM, n_init=NINIT)
            for _ in range(n_null)]
    return _z(obs, null)


def z_regions(X, region, rng, n_null=N_NULL):
    """Regions compared with Procrustes, then clustered in region space."""
    D = sm.distance_matrix(X, region, n_pcs=N_PCS, alpha=1)
    E = sm.classical_mds(D, EMB_DIM)
    obs = sm.best_silhouette(E, KRANGE)
    null = [sm.best_silhouette(sm.curve_gaussian_null(E, rng), KRANGE)
            for _ in range(n_null)]
    return _z(obs, null)


def calibrate(psi_scales, seeds=(0, 1, 2), a=0.0, n_neurons=N_NEURONS,
              n_regions=N_REGIONS):
    """Pick PSI_SCALE: the c = 0 column must sit at z ~ 0 (a continuum is not
    lumpy) while c = 1 separates.  Returns {scale: (z at c=0, z at c=1)}."""
    out = {}
    for ps in psi_scales:
        z0, z1 = [], []
        for s in seeds:
            for c, acc in ((0.0, z0), (1.0, z1)):
                X, region, _ = field(a, c, seed=s, psi_scale=ps,
                                     n_neurons=n_neurons, n_regions=n_regions)
                acc.append(z_regions(X, region, np.random.default_rng(1000 + s)))
        out[ps] = (float(np.mean(z0)), float(np.mean(z1)))
    return out


# --------------------------------------------------- the decoding readout
# `z_regions` compares whole region populations with a shape distance.  This asks
# the pairwise question the other way round -- given ONE neuron, which of these two
# regions is it from? -- and then runs the identical embedding and clustering test,
# so the two are comparable cell by cell over the grid.
#
# QDA and not logistic regression, and that is forced rather than chosen.
# `R.region_weights` draws the long-axis coordinate from a symmetric +-a*sigma
# mixture and the short-axis one from a zero-mean Gaussian, so E[W] = 0 for EVERY
# region at every (a, c).  All regions share one mean tuning curve and a linear
# decoder is at chance over the whole grid by construction -- verified on a spot
# check, not assumed.  Only a decoder that reads the second moment can see
# anything here, which is also what a Procrustes comparison of regions sees.
def shape_coords(X):
    """Each neuron's (skew, width) in the generator's own frame."""
    return (X - R.MEAN) @ R.MODES.T / R.SCALE


def decoding_dissimilarity(A):
    """Accuracy -> dissimilarity anchored at chance, with a zero diagonal.

    An accuracy matrix is not one already: d(i,i) = 0 while two IDENTICAL regions
    score 0.5, and MDS would read that step as structure.
    """
    D = np.clip(A - 0.5, 0.0, None)
    np.fill_diagonal(D, 0.0)
    return D


def _qda():
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    return QuadraticDiscriminantAnalysis(reg_param=0.05)


def z_decode(X, region, rng, n_null=N_NULL, model=_qda):
    """Regions compared by pairwise decoding, then clustered in region space."""
    A = sm.decoding_matrix(shape_coords(X), region, model=model())
    E = sm.classical_mds(decoding_dissimilarity(A), EMB_DIM)
    obs = sm.best_silhouette(E, KRANGE)
    null = [sm.best_silhouette(sm.curve_gaussian_null(E, rng), KRANGE)
            for _ in range(n_null)]
    return _z(obs, null)
