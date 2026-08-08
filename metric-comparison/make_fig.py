"""Figure: clustering and behavioral regression require a proper metric.

Run analyses.py first (writes results.npz), then:

    python make_fig.py            # figure.pdf  (composed, ready to drop in)
    python make_fig.py --panels   # also A.pdf .. E.pdf, one per panel

Each panel is drawn into a matplotlib SubFigure, so the same code produces both
the composed figure and the standalone panels.
"""
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

import metrics as mx
from analyses import cluster_ari, knn_r2
from simulate_data import THETAS, simulate

mpl.rcParams.update({"text.usetex": False, "svg.fonttype": "none",
                     "pdf.fonttype": 42, "font.size": 7,
                     "axes.titlesize": 7.5, "axes.linewidth": 0.7})

PROC = "#0b0b0b"        # Procrustes   -- a metric
PRED = "#eb6834"        # predictivity -- not a metric
PRED2 = "#f0a68a"       # predictivity, transposed
SYM = "#8a8984"         # symmetrized control
TYPE_COLORS = ["#2a78d6", "#1baf7a", "#eda100"]
# Richness of the individuals shown in panel a.  Matched across types so the
# panel isolates tuning width; 0 keeps the curves legible.
EXAMPLE_RICHNESS = 0
NAMES = ["Proc", "$D$", "$D^\\mathsf{T}$", "sym"]
COLORS = [PROC, PRED, PRED2, SYM]

# ---------------------------------------------------------------- load
R = np.load("results.npz")
V = list(R["variants"])
ari, knn, tri, neg = R["ari"], R["knn"], R["tri"], R["negeig"]
ip, ir, it, isym = (V.index(v) for v in
                    ("procrustes", "predictivity", "predictivity_T",
                     "predictivity_sym"))
N_SEEDS = len(ari[ip])

Xs, behavior, labels, weights, richness = simulate(seed=0)
Ps = mx.to_pcs(Xs)
Dp = mx.pairwise_procrustes(Ps)
Dr = mx.pairwise_predictivity(Ps, seed=0)
Ds = mx.symmetrize(Dr)
K = len(Dp)


def dist_plot(ax, data):
    """Violin per variant, always with a visible mean bar -- the Procrustes
    distributions are near-degenerate (ARI 1.000 +- 0.000)."""
    for k, (vals, c) in enumerate(zip(data, COLORS), start=1):
        if np.std(vals) > 1e-6:
            body = ax.violinplot([vals], positions=[k], showextrema=False,
                                 widths=0.8)["bodies"][0]
            body.set_facecolor(c)
            body.set_alpha(0.8)
        ax.plot([k - 0.3, k + 0.3], [np.mean(vals)] * 2, "-", lw=1.8, color=c)
    ax.set_xticks(range(1, 5), NAMES, rotation=45)
    ax.set_xlim(0.4, 4.6)


# ---------------------------------------------------------------- panels
def panel_a(sf):
    """The cohort: tuning curves, manifolds and behavior, one per type."""
    gs = sf.add_gridspec(2, 3)
    # Match the examples on richness.  Richness is drawn independently of type,
    # but it adds jitter when the response is plotted against theta alone (the
    # extra variables vary independently across conditions).  Picking whichever
    # individual came first would suggest a link between broad tuning and noisy
    # curves that does not exist.
    target = EXAMPLE_RICHNESS
    examples = [np.where(labels == t)[0][np.argmin(
        np.abs(richness[labels == t] - target))] for t in range(3)]
    titles = ["broad tuning", "intermediate", "sharp tuning"]

    for col, (idx, ttl) in enumerate(zip(examples, titles)):
        ax = sf.add_subplot(gs[0, col])
        for n in range(0, 60, 8):
            ax.plot(THETAS, Xs[idx][n], lw=0.6, color=TYPE_COLORS[col], alpha=0.7)
        ax.set_title(f"{ttl}\nbehavior = {behavior[idx]:.2f}", color=TYPE_COLORS[col])
        ax.set_xticks([-np.pi, 0, np.pi], ["$-\\pi$", "0", "$\\pi$"])
        ax.set_yticks([])
        ax.set_xlabel(r"$\theta$", labelpad=0)
        if col == 0:
            ax.set_ylabel("firing rate")
        sns.despine(ax=ax, left=True)

        ax = sf.add_subplot(gs[1, col], projection="3d")
        pcs = PCA(3).fit_transform(Xs[idx].T)
        ax.plot(pcs[:, 0], pcs[:, 1], pcs[:, 2], "-", lw=1.2, color=TYPE_COLORS[col])
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            a.set_ticklabels([])
            a.set_ticks([])
        ax.grid(False)
        if col == 0:
            ax.text2D(-0.02, 0.06, "PC space", transform=ax.transAxes,
                      fontsize=6, color="0.35")


def panel_b(sf):
    """Asymmetry: d(i->j) against d(j->i)."""
    axs = sf.subplots(1, 2)
    iu = np.triu_indices(K, 1)
    for ax, D, c, ttl in [(axs[0], Dp, PROC, "Procrustes"),
                          (axs[1], Dr, PRED, "predictivity, $D$")]:
        hi = D.max() * 1.05
        ax.plot([0, hi], [0, hi], "-", lw=0.7, color="0.7", zorder=0)
        ax.plot(D[iu], D.T[iu], "o", ms=2.5, color=c, alpha=0.6, mec="none")
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("$d(i \\rightarrow j)$")
        ax.set_title(f"{ttl}\nasymmetry = {mx.asymmetry(D):.2f}", color=c)
        sns.despine(ax=ax)
    axs[0].set_ylabel("$d(j \\rightarrow i)$")
    ep = R["extreme_pair"]
    axs[1].text(0.04, 0.95, f"constructed pair:\n$R^2$ = {ep[0]:.2f} one way\n"
                            f"$R^2$ = {ep[1]:.2f} the other",
                transform=axs[1].transAxes, va="top", fontsize=5.5, color=PRED)


def panel_c(sf):
    """Clustering: the same data in two orientations, two different answers.

    Matrices rather than dendrograms on purpose -- scipy.linkage needs a
    symmetric input, and symmetrize(D) == symmetrize(D.T), which would hide
    exactly the order-dependence this panel exists to show.
    """
    gs = sf.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.25])
    srt = np.argsort(labels, kind="stable")
    bounds = np.cumsum(np.bincount(labels))[:-1]

    for k, (D, c, ttl) in enumerate([(Dp, PROC, "Procrustes"),
                                     (Dr, PRED, "predictivity, $D$"),
                                     (Dr.T, PRED2, "predictivity, $D^\\mathsf{T}$")]):
        ax = sf.add_subplot(gs[0, k])
        ax.imshow(D[np.ix_(srt, srt)], cmap="magma_r", interpolation="nearest")
        for b in bounds:
            ax.axhline(b - 0.5, color="w", lw=0.8)
            ax.axvline(b - 0.5, color="w", lw=0.8)
        assign = AgglomerativeClustering(n_clusters=3, metric="precomputed",
                                         linkage="average").fit_predict(D)[srt]
        truth = labels[srt]
        remap = {a: np.bincount(truth[assign == a], minlength=3).argmax()
                 for a in np.unique(assign)}
        for x, a in enumerate(assign):
            ax.plot(x, K + 0.8, "s", ms=2.4, color=TYPE_COLORS[remap[a]],
                    clip_on=False)
        ax.set_title(f"{ttl}\nARI = {cluster_ari(D, labels):.2f}", color=c)
        ax.set_xticks([])
        ax.set_yticks([])
        if k == 0:
            ax.set_ylabel("individuals\n(sorted by type)")

    ax = sf.add_subplot(gs[0, 3])
    dist_plot(ax, [ari[ip], ari[ir], ari[it], ari[isym]])
    ax.set_ylabel("ARI vs ground truth")
    ax.set_title(f"{N_SEEDS} seeds")
    ax.set_ylim(-0.1, 1.05)
    sns.despine(ax=ax)


def panel_d(sf, scatters=None, show_dist=True):
    """Predicting behavior from position in the space.

    `scatters` selects which dissimilarities get a predicted-vs-true scatter.
    `show_dist` adds the across-seed distribution for all four variants.
    """
    if scatters is None:
        scatters = [(Dp, PROC, "Procrustes"), (Dr, PRED, "predictivity, $D$")]
    n = len(scatters)
    ratios = [1] * n + ([1.15] if show_dist else [])
    axs = np.atleast_1d(sf.subplots(1, n + (1 if show_dist else 0),
                                    width_ratios=ratios))
    lim = (behavior.min() - 0.4, behavior.max() + 0.4)
    for ax, (D, c, ttl) in zip(axs[:n], scatters):
        masked = D + np.eye(K) * 1e9
        pred = np.array([behavior[np.argsort(masked[i])[:3]].mean()
                         for i in range(K)])
        ax.plot(lim, lim, "-", lw=0.7, color="0.7", zorder=0)
        ax.scatter(behavior, pred, s=10, c=[TYPE_COLORS[t] for t in labels],
                   edgecolors="none")
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect("equal")
        ax.set_xlabel("true behavior")
        ax.set_title(f"{ttl}\n$R^2$ = {knn_r2(D, behavior):.2f}", color=c)
        sns.despine(ax=ax)
    axs[0].set_ylabel("predicted (kNN, $k$=3)")
    if not show_dist:
        return

    ax = axs[n]
    dist_plot(ax, [knn[ip], knn[ir], knn[it], knn[isym]])
    ax.axhline(0, ls=":", lw=0.7, color="0.4")
    ax.text(0.55, 0.06, "cohort mean", fontsize=5, color="0.4", ha="left")
    ax.set_ylabel("$R^2$")
    ax.set_title(f"{N_SEEDS} seeds")
    sns.despine(ax=ax)


def panel_e(sf):
    """Why it fails: the metric axioms."""
    axs = sf.subplots(1, 2)
    order = [ip, ir, it, isym]
    for ax, vals, ylab in [(axs[0], 100 * tri, "triangle\nviolations (%)"),
                           (axs[1], 100 * neg, "negative\neigenvalue mass (%)")]:
        ax.bar(range(4), [vals[i].mean() for i in order],
               yerr=[vals[i].std() for i in order],
               color=COLORS, error_kw=dict(lw=0.7))
        ax.set_xticks(range(4), NAMES, rotation=45)
        ax.set_ylabel(ylab)
        sns.despine(ax=ax)
    axs[0].text(0, 0.6, "0%", ha="center", fontsize=5.5, color=PROC)
    axs[1].text(0, 1.2, "0%", ha="center", fontsize=5.5, color=PROC)


PANELS = [("a", panel_a, (4.6, 2.9)),
          ("b", panel_b, (3.4, 1.9)),
          ("c", panel_c, (5.6, 2.1)),
          ("d", panel_d, (4.8, 1.9)),
          ("e", panel_e, (3.0, 1.9))]


def tag(sf, letter):
    sf.text(0.005, 0.99, letter, fontsize=10, fontweight="bold",
            va="top", ha="left")


# ---------------------------------------------------------------- composed
def make_figure(path="figure.pdf"):
    fig = plt.figure(figsize=(7.2, 9.2), layout="constrained")
    rows = fig.subfigures(4, 1, height_ratios=[2.95, 2.15, 2.15, 2.05])
    top = rows[1].subfigures(1, 2, width_ratios=[3.3, 3.1])

    for sf, (letter, fn, _) in zip([rows[0], top[0], rows[2], rows[3], top[1]],
                                   [PANELS[0], PANELS[1], PANELS[2],
                                    PANELS[3], PANELS[4]]):
        fn(sf)
        tag(sf, letter)

    fig.savefig(path, transparent=True)
    plt.close(fig)
    print("wrote", path)


def make_figure_simple(path="figure_simple.pdf"):
    """Stripped-down version: the cohort, then behavioral regression using the
    symmetrized predictivity -- i.e. the steelmanned form of the comparison,
    after the asymmetry objection has already been conceded."""
    fig = plt.figure(figsize=(6.4, 5.0), layout="constrained")
    rows = fig.subfigures(2, 1, height_ratios=[2.9, 2.0])

    panel_a(rows[0])
    tag(rows[0], "a")
    panel_d(rows[1], show_dist=False,
            scatters=[(Dp, PROC, "Procrustes"),
                      (Dr, PRED, "predictivity, $D$"),
                      (Ds, SYM, "predictivity, symmetrized")])
    tag(rows[1], "b")

    fig.savefig(path, transparent=True)
    plt.close(fig)
    print("wrote", path)


def make_panels():
    for letter, fn, size in PANELS:
        fig = plt.figure(figsize=size, layout="constrained")
        fn(fig.subfigures(1, 1))
        name = f"{letter.upper()}.pdf"
        fig.savefig(name, transparent=True)
        plt.close(fig)
        print("wrote", name)


if __name__ == "__main__":
    if "--simple" in sys.argv:
        make_figure_simple()
    else:
        make_figure()
        if "--panels" in sys.argv:
            make_panels()

    print(f"\nnumbers in the figure ({N_SEEDS} seeds):")
    for i, v in zip([ip, ir, it, isym], V):
        print(f"  {v:<18} ARI {ari[i].mean():.3f}  kNN {knn[i].mean():+.3f}  "
              f"tri {100*tri[i].mean():.1f}%  neg.eig {100*neg[i].mean():.1f}%")
