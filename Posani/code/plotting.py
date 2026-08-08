"""Plots for the region-to-region shape analysis."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

import data as D_

# Matched to the paper's figures (netrep-journal-figure-4.pdf): Arial, ~8 pt text,
# square data panels of about 3.4 in inside a 8.43 in wide figure.
# fonttype 42 embeds TrueType, so the text stays editable text in Affinity
# (matplotlib's Type 3 default arrives there as unselectable outlines).
PAGE_WIDTH = 8.43    # in, the width of netrep-journal-figure-4.pdf
PAGE_HEIGHT = 4.20   # in, its height -- match this to sit beside it, not below
PANEL = PAGE_WIDTH / 2   # inches per panel cell, so a two-column figure is page width
PANEL_LETTER = 16.6  # pt, the size of the a/b/c labels in the paper

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": .8,
    "xtick.major.width": .8,
    "ytick.major.width": .8,
})


def figsize(ncols=1, nrows=1, panel=PANEL):
    """Figure size at the paper's panel scale.

    Two columns come out exactly PAGE_WIDTH, so the PDF drops in beside
    netrep-journal-figure-4.pdf at 100% -- no rescaling.
    """
    return (panel * ncols, panel * nrows)


def panel_letter(ax, letter, x=-.22, y=1.06):
    """The a/b/c label, at the paper's size."""
    return ax.text(x, y, letter, transform=ax.transAxes, fontsize=PANEL_LETTER,
                   fontweight="bold", va="top", ha="left")

PALETTE = dict(zip(D_.MODULES, plt.cm.tab10.colors))
PALETTE["other"] = (0.6, 0.6, 0.6)


def save(fig, name, folder="figures/notebook", crop=False):
    """Write an editable vector PDF (plus a PNG for quick viewing).

    The canvas is left at exactly the requested figsize -- `bbox_inches="tight"`
    would crop it to the ink and the page would no longer match the paper's
    dimensions. Pass crop=True if you want the trimmed version instead.
    """
    import os
    os.makedirs(folder, exist_ok=True)
    bbox = "tight" if crop else None
    fig.savefig(f"{folder}/{name}.pdf", bbox_inches=bbox)
    fig.savefig(f"{folder}/{name}.png", dpi=200, bbox_inches=bbox)
    return f"{folder}/{name}.pdf"


def colors(data):
    return np.array([PALETTE[m] for m in data.modules()])


def fit_line(ax, x, y, color="k", lw=1.4, alpha=.8):
    """Least-squares line through a cloud of points, spanning the observed range."""
    b = np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)
    xs = np.array([np.min(x), np.max(x)])
    ax.plot(xs, np.polyval(b, xs), color=color, lw=lw, alpha=alpha, zorder=1)
    return ax


def cluster_order(D):
    """Ward linkage of the dissimilarity matrix, with optimal leaf ordering."""
    v = squareform(D, checks=False)
    link = hierarchy.ward(v)
    return hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(link, v)), link


def distmat(D, data, ax=None, cmap="magma_r", fontsize=None):
    fontsize = plt.rcParams["xtick.labelsize"] if fontsize is None else fontsize
    ax = ax or plt.subplots(figsize=figsize())[1]
    order, _ = cluster_order(D)
    off = D[np.triu_indices(len(D), 1)]
    im = ax.imshow(D[order][:, order], cmap=cmap, clim=(off.min(), off.max()))
    c = colors(data)
    ax.set_xticks(range(len(D)))
    ax.set_yticks(range(len(D)))
    ax.set_xticklabels(data.regions[order], rotation=90, fontsize=fontsize)
    ax.set_yticklabels(data.regions[order], fontsize=fontsize)
    for t, i in zip(ax.get_xticklabels(), order):
        t.set_color(c[i])
    for t, i in zip(ax.get_yticklabels(), order):
        t.set_color(c[i])
    cb = plt.colorbar(im, ax=ax, shrink=.7)
    cb.set_label("Procrustes distance", fontsize=plt.rcParams["axes.labelsize"])
    cb.ax.tick_params(labelsize=fontsize)
    ax.set_box_aspect(1)
    return ax


def mds(xy, var, data, ax=None, by="module", size=70, fontsize=None):
    """Scatter of the embedded regions, coloured by module or by hierarchy."""
    fontsize = plt.rcParams["font.size"] if fontsize is None else fontsize
    ax = ax or plt.subplots(figsize=figsize())[1]
    x, y = xy.T
    if by == "module":
        for m in dict.fromkeys(data.modules()):
            s = data.modules() == m
            ax.scatter(x[s], y[s], color=PALETTE[m], s=size, lw=0, label=m)
        ax.legend(fontsize=fontsize - 1, frameon=False)
    else:
        sc = ax.scatter(x, y, c=data.hierarchy(), cmap="viridis", s=size, lw=0)
        cb = plt.colorbar(sc, ax=ax, fraction=.046, pad=.04)
        cb.set_label("position in cortical hierarchy", fontsize=fontsize)
        cb.ax.tick_params(labelsize=fontsize - 1)
    for _x, _y, a in zip(x, y, data.regions):
        # offset in points, so the label clears the marker whatever its size
        ax.annotate(a, (_x, _y), textcoords="offset points",
                    xytext=(.5 * np.sqrt(size), .4 * np.sqrt(size)), fontsize=fontsize)
    ax.set_xlabel(f"PC1 ({var[0]:.0%})", fontsize=fontsize)
    ax.set_ylabel(f"PC2 ({var[1]:.0%})", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 1)
    ax.margins(.13)              # room for the offset labels near the edges
    ax.set_box_aspect(1)
    return ax


def regression_axis(ax, coef, xy, n_iso=5, color="k", label=None):
    """Draw a 2D regression as an axis in the embedding plane.

    `coef` are the weights on the two coordinates. The arrow points along the
    gradient, i.e. the direction in which the predicted value grows fastest, and the
    dotted lines are its level sets -- points on one of them get the same prediction.
    """
    u = np.asarray(coef, float)[:2]
    u = u / np.linalg.norm(u)
    v = np.array([-u[1], u[0]])                 # runs along the level sets
    c = xy.mean(0)
    t = (xy - c) @ u
    lo, hi = t.min() * 1.15, t.max() * 1.15
    ax.annotate("", xy=c + hi * u, xytext=c + lo * u, zorder=0,
                arrowprops=dict(arrowstyle="-|>", lw=1.4, color=color, alpha=.55))
    if label:
        ax.text(*(c + hi * u), f" {label}", fontsize=8, color=color,
                ha="left", va="center", zorder=0)
    w = np.abs((xy - c) @ v).max() * 1.15
    for s in np.linspace(lo, hi, n_iso):
        m = c + s * u
        ax.plot([m[0] - w * v[0], m[0] + w * v[0]],
                [m[1] - w * v[1], m[1] + w * v[1]],
                color=color, lw=.5, ls=":", alpha=.35, zorder=0)
    return ax


def prediction(y, preds, names, ax=None, xlabel="true", ylabel="predicted",
               title=None, point_labels=None, size=45, fontsize=None, line=True):
    """Predicted vs true, one colour per model (e.g. in-sample and cross-validated)."""
    fontsize = plt.rcParams["font.size"] if fontsize is None else fontsize
    ax = ax or plt.subplots(figsize=figsize())[1]
    for pred, name, col in zip(preds, names, ["C1", "C3", "C0"]):
        ax.scatter(y, pred, s=size, color=col, alpha=.75, lw=0, label=name)
        if line:
            fit_line(ax, y, pred, color=col)
    lim = [min(np.min(y), *[p.min() for p in preds]),
           max(np.max(y), *[p.max() for p in preds])]
    ax.plot(lim, lim, "k--", lw=1, zorder=0)
    if point_labels is not None:
        off = (.5 * np.sqrt(size), .4 * np.sqrt(size))
        for _y, _p, t in zip(y, preds[-1], point_labels):
            ax.annotate(t, (_y, _p), textcoords="offset points", xytext=off,
                        fontsize=fontsize - 1)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 1)
    ax.legend(fontsize=fontsize - 2, frameon=False)
    if title:
        ax.set_title(title, fontsize=fontsize)
    ax.margins(.10)              # room for the offset labels near the edges
    ax.set_box_aspect(1)
    return ax


def scatter(x, y, xlabel, ylabel, ax=None, identity=False, title=None,
            size=18, fontsize=None, line=True, color="C3"):
    fontsize = plt.rcParams["font.size"] if fontsize is None else fontsize
    ax = ax or plt.subplots(figsize=figsize())[1]
    ax.scatter(x, y, s=size, color=color, alpha=.55, lw=0)
    if line:
        fit_line(ax, x, y)
    if identity:
        lim = [min(np.min(x), np.min(y)), max(np.max(x), np.max(y))]
        ax.plot(lim, lim, "k--", lw=1, zorder=0)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 1)
    if title:
        ax.set_title(title, fontsize=fontsize)
    ax.set_box_aspect(1)
    return ax
