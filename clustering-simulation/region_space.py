"""Categoricality in neuron space says nothing about whether regions form types.

Three scenarios, one generator, one figure.  A region's neurons are an elongated
cloud in tuning-shape space: `amount` (a) sets how bimodal it is along its long
axis (how categorical the region is), `kind` (psi) sets that axis's orientation
(what the region is categorical *about*).

The paper calls these three fields "gradient", "unstructured" and "clustered";
the scenario keys below are the older names and mean the same things.

  SCENARIO 1  "continuum"  (paper: GRADIENT)
      regions vary continuously: `amount` sweeps AMOUNT_MIN..AMOUNT_MAX and
      `kind` sweeps with it.  So the distribution of per-region categoricality is
      SPREAD OUT -- some regions are strongly categorical, some not at all -- and
      in region space the regions lie on a continuum with no types in it.

  SCENARIO 2  "no_gradient"  (paper: UNSTRUCTURED)
      the same set of `amount` values and the same set of `kind` values as
      scenario 1, but the pairing between them is permuted away.  The region
      cloud and the categoricality distribution are therefore both unchanged;
      only the gradient -- the correspondence between the two -- is destroyed.

  SCENARIO 3  "kinds"  (paper: CLUSTERED)
      no region is categorical at all (`amount` = 0, so every per-region z sits at
      zero), but `kind` takes THREE discrete values, so region space contains
      three clean clusters.

Read the first column of the figure and you learn nothing about the third.


THE TWO KNOBS ARE ORTHOGONAL BY CONSTRUCTION.

  amount (a)   HOW MUCH structure the region has.  Along its long axis the
               neurons are a mixture of two components at +-a*SIG_L with spread
               sqrt(1-a^2)*SIG_L, so the variance along that axis is SIG_L^2 at
               EVERY setting.  a = 0 is one continuous elongated cloud; a = 1 is
               two tight clusters.  This is what `categoricality` measures.

  kind (psi)   WHAT the structure is: the orientation of that axis in shape space.
               Same spread, same tightness -- different identity.

Because the variance along each axis is fixed, the second moment of a region's
weights, SIG_L^2 uu' + SIG_S^2 vv', depends on psi and NOT on a.  The two knobs
are therefore visible to different things, which is the whole point.

THE SHAPE SPACE IS BUILT SO `kind` IS A MIRROR SYMMETRY.  The bank of tuning
curves is skewed (`exp(k(cos t - 1)) * (1 + s sin t)`), which gives one ODD and
one EVEN shape mode.  The map theta -> -theta therefore flips the first weight and
fixes the second, sending psi -> -psi.  A region at -psi is thus the exact
theta-mirror image of one at +psi, so:

  * any statistic invariant to permuting the theta bins -- which the
    categoricality score is, since it depends only on Euclidean distances between
    standardised tuning curves -- is IDENTICALLY DISTRIBUTED for the two.
    It is provably blind to `kind`, not merely underpowered.
  * Procrustes compares `bins x components` matrices up to an orthogonal
    transform of the COMPONENTS, not of the bins, so it is not invariant to that
    mirror and can see `kind`.

Two earlier parameterisations of the shape space failed and are worth recording:
  * `cos(theta + s sin theta)` is even in theta for every s, so it produces no
    skew at all and both modes come out even.
  * Scaling the two modes by their own standard deviations, or defining them in
    raw rather than standardised curve space, makes rotating psi change a
    region's amplitude -- which the pipeline's per-neuron standardisation then
    treats differently, so the two knobs stop being orthogonal.


TWO THINGS ABOUT SCENARIO 1 THAT ARE NOT ARBITRARY.

First, its regions are SAMPLED along the gradient, not evenly spaced.  An evenly
spaced lattice of regions is itself non-Gaussian, and `continuum_or_types` would
report that regularity as structure.

Second, it sweeps only PSI_SPAN = 0.3 of the psi range.  Procrustes distance is a
nonlinear function of psi, so a wide sweep puts the regions on a strongly CURVED
arc -- and the silhouette-versus-Gaussian test detects departure from a Gaussian,
which curvature is.  Over the full range this simulation returns p = 0.01 for a
continuum containing no types whatsoever, and p = 0.08-0.19 at PSI_SPAN = 0.5.
That is a limitation of the test, not evidence of types, and it is worth knowing:
a strongly curved continuum of regions can be reported as lumpy.  At
PSI_SPAN = 0.3 the continuum is close to straight and the test correctly returns
p = 0.19-0.38 across seeds, while region PC1 still recovers position along the
continuum at |rho| = 0.93-0.97.

Note `amount` and `kind` are swept together in scenario 1 on purpose.  The
generator holds a region's total spread fixed as `amount` changes, so the second
moment of its weights -- which is what a Procrustes comparison of regions sees --
depends on `kind` and not on `amount`.  Sweeping `amount` alone would spread the
categoricality histogram while leaving region space almost unchanged; sweeping
`kind` alone would give the continuum but a flat histogram.  Scenario 1 needs
both, and they are two visible consequences of one underlying gradient.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import shapemetrics as sm                      # noqa: E402

# ------------------------------------------------------- the simulated field
N_REGIONS, N_NEURONS = 60, 100
N_KINDS = 3                        # scenario 2
AMOUNT_MIN, AMOUNT_MAX = 0.40, 0.95   # scenario 1 sweeps this range
PSI = np.pi / 4                    # the extreme of the kind axis
PSI_SPAN = 0.3                     # scenario 1 sweeps part of the psi range

# ------------------------------------------------------------- a region's cloud
N_THETAS = 120
SIG_L, SIG_S = 1.0, 0.30           # spread along a region's long / short axis
NOISE = 0.004
THETAS = np.linspace(-np.pi, np.pi, N_THETAS, endpoint=False)

# ------------------------------------------------------------------ the tests
N_PCS = 10                         # per-region PCs handed to Procrustes
KLIM, NINIT = (2, 11), 10          # k swept when clustering NEURONS
KMAX_REGION = 6                    # k swept when clustering REGIONS
EMB_DIM = 5                        # dimensions of the region embedding


def _shape_space():
    """Mean and two modes of a bank of SKEWED von Mises curves, in STANDARDISED
    curve space (where the clustering pipeline lives).  One mode is odd in theta
    and one is even, which is what makes `kind` a mirror symmetry."""
    bank = np.stack([np.exp(k * (np.cos(THETAS) - 1)) * (1 + s * np.sin(THETAS))
                     for k in np.linspace(1.5, 7, 40)
                     for s in np.linspace(-0.8, 0.8, 25)])
    bz = (bank - bank.mean(1, keepdims=True)) / bank.std(1, keepdims=True)
    p = PCA(2).fit(bz)
    # one common scale for both modes: see the module docstring
    return p.mean_, p.components_, float(np.sqrt(p.explained_variance_).mean())


MEAN, MODES, SCALE = _shape_space()


# ---------------------------------------------------------------- generator
def region_weights(amount, psi, rng, n_neurons=N_NEURONS):
    u = np.array([np.cos(psi), np.sin(psi)])
    v = np.array([-np.sin(psi), np.cos(psi)])
    sgn = rng.choice([-1.0, 1.0], n_neurons)
    t = amount * SIG_L * sgn + np.sqrt(max(1 - amount ** 2, 0)) * SIG_L \
        * rng.standard_normal(n_neurons)
    s = SIG_S * rng.standard_normal(n_neurons)
    return t[:, None] * u + s[:, None] * v


def simulate(amounts, psis, seed=0, n_neurons=N_NEURONS):
    rng = np.random.default_rng(seed)
    W = np.concatenate([region_weights(a, p, rng, n_neurons)
                        for a, p in zip(amounts, psis)])
    X = MEAN + SCALE * W @ MODES
    return (X + NOISE * rng.standard_normal(X.shape),
            np.repeat(np.arange(len(amounts)), n_neurons))


def scenario(name, seed=0, n_regions=N_REGIONS):
    """Return (X, region, colour); `colour` is the latent that orders or groups
    the regions -- position on the continuum, or which kind."""
    if name == "continuum":
        # regions are SAMPLED along the gradient rather than evenly spaced: an
        # evenly spaced lattice is itself non-Gaussian, and the region-space test
        # would flag that regularity as structure
        t = np.sort(np.random.default_rng(seed).uniform(0, 1, n_regions))
        amounts = AMOUNT_MIN + (AMOUNT_MAX - AMOUNT_MIN) * t
        psis = (-PSI + 2 * PSI * t) * PSI_SPAN
        colour = t
    elif name == "no_gradient":
        # identical to "continuum" except that `a` is permuted across regions.
        # The set of psi values and the set of a values are both unchanged, so the
        # region cloud and the categoricality distribution are the same; only the
        # correspondence between them -- the gradient -- is destroyed.
        rng = np.random.default_rng(seed)
        t = np.sort(rng.uniform(0, 1, n_regions))
        amounts = rng.permutation(AMOUNT_MIN + (AMOUNT_MAX - AMOUNT_MIN) * t)
        psis = (-PSI + 2 * PSI * t) * PSI_SPAN
        colour = t
    elif name == "kinds":
        # NO region is categorical: `a` = 0 throughout, so every per-region z
        # sits at zero.  All the structure is in `kind`, which takes N_KINDS
        # discrete values -- so the field is unambiguously clustered in region
        # space while containing not one clustered region.  That is the whole
        # claim of the panel, and setting `a` = 0 states it in its strongest
        # form.
        #
        # `a` is free here because it does not touch the region geometry: it
        # leaves the second moment of a region's weights untouched, and it is
        # the second moment that Procrustes compares.  Verified directly --
        # against `a` drawn over [AMOUNT_MIN, AMOUNT_MAX] as the other two
        # scenarios do, the region-space result is unchanged across seeds 0-2:
        # silhouette 0.725-0.737 vs 0.721-0.761, z +8.6/+10.1 vs +9.8/+10.7,
        # p = 0.005 (the floor, 200 null draws) in all six runs, and
        # between/within-kind separation 6.0-6.4 vs 5.8-7.0.
        amounts = np.zeros(n_regions)
        psis = np.tile(np.linspace(-PSI, PSI, N_KINDS), n_regions // N_KINDS)
        colour = np.tile(np.arange(N_KINDS), n_regions // N_KINDS)
    else:
        raise ValueError(name)
    X, region = simulate(amounts, psis, seed=seed, n_neurons=N_NEURONS)
    return X, region, colour


# --------------------------------------------------------------- neuron space
def categoricality(X, region, rng, n_draws=8):
    """Per-region z: is this region's own cloud lumpier than one continuous
    cloud?  Each region against a Gaussian matched to itself -- drawn in raw
    curve space and pushed through the identical pipeline, so the pipeline's
    nonlinearity hits data and null alike.  The first column of the figure."""
    out = []
    for r in np.unique(region):
        A = X[region == r]
        obs = sm.pipeline_silhouette(A, k_lim=KLIM, n_init=NINIT)
        nl = np.array([sm.pipeline_silhouette(sm.curve_gaussian_null(A, rng),
                                              k_lim=KLIM, n_init=NINIT)
                       for _ in range(n_draws)])
        out.append((obs - nl.mean()) / (nl.std() + 1e-12))
    return np.array(out)


# --------------------------------------------------------------- region space
def procrustes_distances(X, region, n_pcs=N_PCS):
    """Each region's whole population, compared with the shape metric."""
    return sm.distance_matrix(X, region, n_pcs=n_pcs, alpha=1)


def pca_embed(D, n_components=EMB_DIM):
    """Classical MDS: deterministic, and its axes come out ordered by variance --
    which matters here, where PC1 is read as position along the continuum."""
    return sm.classical_mds(D, n_components)


def mean_embed(X, region, n_components=2):
    """The averaging alternative: collapse each region to the MEAN of its
    neurons' standardised tuning vectors, then PCA across regions.

    This also puts every region in one common space -- it is the analysis of
    Posani et al.'s Fig. 2c -- but the space is built from an average over
    neurons, so it keeps only the first moment and discards the population's
    shape.  Cheap: no pairwise Procrustes, just one average per region.
    """
    Xs = (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
    M = np.stack([Xs[region == r].mean(0) for r in np.unique(region)])
    return PCA(n_components).fit_transform(M)


def best_silhouette(E):
    return sm.best_silhouette(E, range(2, KMAX_REGION + 1))


def continuum_or_types(E, n_draws=200, seed=0):
    """Are the regions lumpy, or one continuous cloud?

    Observed best silhouette over k, against Gaussians matched to the region
    embedding's mean and covariance.  A continuum is not lumpy in this sense --
    a line of points and a Gaussian blob score alike -- while genuine types are.
    """
    rng = np.random.default_rng(seed)
    obs = best_silhouette(E)
    nl = np.array([best_silhouette(sm.curve_gaussian_null(E, rng))
                   for _ in range(n_draws)])
    return dict(obs=float(obs), null=nl,
                z=float((obs - nl.mean()) / (nl.std() + 1e-12)),
                p=float((np.sum(nl >= obs) + 1) / (n_draws + 1)))
