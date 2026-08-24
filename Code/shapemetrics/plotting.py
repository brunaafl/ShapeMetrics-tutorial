"""Plots for the region-to-region shape analysis, in the style of figure 1.

The style is figure 1's, taken from `clustering-simulation/region_space.ipynb`
(the cell that writes `results/region_full.pdf`) so the two region figures sit
beside each other without a visible change of hand:

  colour     `NULLC` blue is always a null, `OBS` firebrick always the data, and
             `GREY` is unlabelled reference material.  `KIND_COLORS` is the
             three-hue categorical set, which passes the lightness, chroma and
             CVD checks that notebook documents.
  markers    small, black-edged, `MARKER_EDGE` wide -- figure 1's regions read as
             discrete points rather than as a haze.
  type       ONE scale for the whole figure (`FS_*`), applied by `typeset` in a
             blanket pass at the end rather than per call, which is what keeps
             a multi-panel figure typographically flat.
  panels     `PC` inches square, `set_box_aspect(1)`, `despine`.
  histograms no y axis at all, and a proxy null/data legend.

Only the family differs from a bare matplotlib: figure 1 never overrode it, so
this is DejaVu Sans and not the Arial the earlier version of this module set.
`pdf.fonttype = 42` is kept regardless of family -- it embeds TrueType, so the
text arrives in Affinity as editable text rather than as outlines.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform


# ------------------------------------------------------------------ the palette
OBS, NULLC, GREY = "#B22222", "#1F6FB2", "0.75"
KIND_COLORS = ["#B22222", "#1F6FB2", "#E8A33D"]

# ------------------------------------------------------------------ the geometry
PC = 1.95            # inches per panel, as in figure 1
PANEL = PC           # kept under its old name for callers that use it
PAGE_WIDTH = 8.43    # in, the width the paper's region figure occupies
PANEL_LETTER = 16.6  # pt, the size of the a/b/c labels in the paper

# ------------------------------------------------------------------- the type
# figure 1's single typographic scale: head / title / label / tick / annotation
FS_HEAD, FS_TITLE, FS_LABEL, FS_TICK, FS_ANNOT = 11, 9, 8.5, 8, 8

# ---------------------------------------------------------------- the markers
MARKER_SIZE = 22     # regions and other one-point-per-thing scatters
MARKER_EDGE = 0.3
POINT_SIZE = 6       # dense clouds, e.g. one point per PAIR of regions
POINT_EDGE = 0.25
LW_OBS = 1.8         # the observed-value line through a null
HEADROOM = 1.42      # room above a histogram for its legend

plt.rcParams.update({
    "svg.fonttype": "none",     # editable text in the svg, as in figure 1
    "text.usetex": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": .8,
    "xtick.major.width": .8,
    "ytick.major.width": .8,
})


def figsize(ncols=1, nrows=1, panel=PC):
    """Figure size at figure 1's panel scale."""
    return (panel * ncols, panel * nrows)


def panel_letter(ax, letter, dx=-30, dy=10):
    """The a/b/c label, at the paper's size.

    Offset in POINTS from the axes' top-left corner rather than in axes fractions,
    so the letters line up with each other on the page whatever each panel's size
    -- an axes-fraction offset puts them at a different physical distance in a
    wide panel than in a narrow one.
    """
    return ax.annotate(letter, xy=(0, 1), xycoords="axes fraction",
                       xytext=(dx, dy), textcoords="offset points",
                       fontsize=PANEL_LETTER, fontweight="bold",
                       va="top", ha="left", annotation_clip=False,
                       gid="panel_letter")


#: Colour per cortical module. Populated by `module_palette(modules)`.
#:
#: This was `PALETTE = dict(zip(D_.MODULES, ...))` evaluated at import, i.e. the
#: shared style module imported a DATASET at module scope -- so plotting could
#: not be used without Posani's data.py on sys.path, and neither could move.
#: Figure 2 calls module_palette() once; every other figure ignores it.
PALETTE = {"other": (0.6, 0.6, 0.6)}


def module_palette(modules):
    """Assign a colour to each cortical module and return the palette."""
    PALETTE.update(zip(modules, plt.cm.tab10.colors))
    PALETTE.setdefault("other", (0.6, 0.6, 0.6))
    return PALETTE


def typeset(fig, extra_axes=()):
    """Figure 1's blanket pass: one type scale over every axis of the figure.

    Called once, last, after every panel is drawn. Per-call font sizes therefore
    do not need to be right -- this is what fixes them -- which is the point:
    a figure typeset panel by panel drifts.
    """
    for ax in list(fig.axes) + list(extra_axes):
        ax.title.set_fontsize(FS_TITLE)
        ax.xaxis.label.set_fontsize(FS_LABEL)
        ax.yaxis.label.set_fontsize(FS_LABEL)
        # an axis that asked for its own tick size keeps it: the matrix carries 22
        # region names per side, which at FS_TICK collide with each other
        ax.tick_params(axis="both", labelsize=getattr(ax, "_tick_fontsize", FS_TICK))
        for t in ax.texts:
            if t.get_gid() not in ("panel_letter", "axis_key", "point_label",
                                   "bar_label"):
                # the a/b/c labels, the PC key and the point names keep their
                # own sizes -- resizing the last of these would undo the layout
                # `label_points` just solved
                t.set_fontsize(FS_ANNOT)
        lg = ax.get_legend()
        if lg is not None:
            for t in lg.get_texts():
                t.set_fontsize(FS_ANNOT)
    return fig


def save(fig, name, folder=None, crop=True, formats=("svg", "pdf", "png")):
    """Write the figure in every format the paper needs, as figure 1 does.

    `crop` trims the canvas to the ink (`bbox_inches="tight"`), which is what
    figure 1 does; pass crop=False to keep the canvas at exactly `figsize`.
    """
    import os
    if folder is None:                     # was "figures/notebook", i.e. relative
        from . import paths                # to wherever Jupyter happened to start
        folder = str(paths.results())
    os.makedirs(folder, exist_ok=True)
    bbox = "tight" if crop else None
    for e in formats:
        fig.savefig(f"{folder}/{name}.{e}", bbox_inches=bbox,
                    dpi=300 if e == "png" else None)
    return f"{folder}/{name}.pdf"


def colors(data):
    return np.array([PALETTE[m] for m in data.modules()])


def fit_line(ax, x, y, color="0.25", lw=1.2, alpha=.9):
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


# --------------------------------------------------------------- the primitives
def pc_arrows(ax, xl="PC 1", yl="PC 2", x0=0.12, y0=0.12, L=0.16, fs=6):
    """Figure 1's axis key: a pair of orthogonal arrows in the bottom-left corner.

    An embedding's axes carry no units and no scale, so a pair of full axis labels
    spends a lot of the panel saying very little.

    `L` is the arm length as a fraction of the panel WIDTH; the vertical arm is
    scaled by the panel's aspect ratio so that the two arms draw the same length
    on the page. Figure 1 could take equal fractions because its panels are
    square -- on a wide panel that would draw an L with one long arm.

    Tagged `axis_key` so `typeset` leaves these at their own small size.
    """
    ax.figure.canvas.draw()
    bb = ax.get_window_extent()
    Lx, Ly = L, L * bb.width / bb.height
    for dx, dy in ((Lx, 0), (0, Ly)):
        ax.annotate("", xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                    xycoords="axes fraction", textcoords="axes fraction",
                    annotation_clip=False, zorder=6,
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color="0.25",
                                    shrinkA=0, shrinkB=0, mutation_scale=7))
    box = dict(facecolor="white", edgecolor="none", alpha=0.75,
               boxstyle="square,pad=0.08")
    # PC 1 centred under its arm, PC 2 centred to the left of its arm
    ax.text(x0 + Lx / 2, y0 - 0.05, xl, transform=ax.transAxes, fontsize=fs,
            color="0.25", ha="center", va="top", zorder=6, bbox=box, gid="axis_key")
    ax.text(x0 - 0.045, y0 + Ly / 2, yl, transform=ax.transAxes, fontsize=fs,
            color="0.25", ha="right", va="center", rotation=90, zorder=6,
            bbox=box, gid="axis_key")
    return ax


def label_points(ax, x, y, labels, fontsize=6, offset=(3.0, 3.0), n_iter=800,
                 pad=3.0, leader=5.0, step=0.55, spring=0.03, marker_r=5.0,
                 colors=None, crowd_r=42.0, crowd_gain=1.15, crowd_max=4.0,
                 rotate=None, boost=None):
    """Name every point, pushing names off each other and off the markers.

    The region names pile up wherever the embedding is dense -- which is exactly
    where they are most worth reading. Each name is relaxed in DISPLAY space (the
    space in which they actually collide) under three forces: it is pushed out of
    any other name's box, pushed off any marker it covers, and pulled weakly back
    toward its own point so it does not drift somewhere meaningless. Any name that
    ends up more than `leader` points from its dot gets a hairline back to it.

    Every text box is measured ONCE and the relaxation then runs in numpy: the
    box sizes never change, only their positions, so re-measuring each iteration
    buys nothing and costs a matplotlib call per label per step.

    `rotate` and `boost` are the manual overrides: {label: degrees} swings a name
    around its own point, positive counter-clockwise, and {label: factor} sends it
    further out along the same direction. The solver resolves overlaps by the shortest
    move, which is not always the most readable one when three names contend for
    the same gap -- this says which way a particular name should go instead.

    No new dependency: `adjustText` does this better, but is not in the
    environment and one figure does not justify adding it.
    """
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(labels)
    cols = [None] * n if colors is None else list(colors)
    px = fig.dpi / 72.0                       # points -> pixels

    texts = [ax.annotate(str(l), (xi, yi), textcoords="offset points",
                         xytext=tuple(offset), fontsize=fontsize, zorder=5,
                         gid="point_label", color=c)
             for xi, yi, l, c in zip(x, y, labels, cols)]

    # Anything already on the panel that a name must not land on: the PC arrow
    # key, and the title. These do not move, so they enter as fixed boxes.
    fixed = [t.get_window_extent(rend) for t in ax.texts
             if t.get_gid() == "axis_key" and t.get_text()]
    if ax.get_title():
        fixed.append(ax.title.get_window_extent(rend))
    fixed = (np.array([[(b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2,
                        (b.x1 - b.x0) / 2, (b.y1 - b.y0) / 2] for b in fixed])
             if fixed else np.zeros((0, 4)))

    # measured once: the boxes only ever move, they never change size
    ext = np.array([[b.width, b.height] for b in
                    (t.get_window_extent(rend) for t in texts)])
    half = ext / 2.0
    pts = ax.transData.transform(np.c_[x, y])          # markers, in pixels
    # Where each name would like to sit. A name in a crowded neighbourhood is
    # sent OUTWARD, away from the middle of the cloud, and the further out it
    # starts the more room the solver has and the more clearly its leader reads.
    # An isolated name keeps the plain up-and-right offset and needs no leader.
    base = np.asarray(offset, float) * px
    reach = np.hypot(*base)
    crowd = (np.hypot(*(pts[:, None, :] - pts[None, :, :]).T).T < crowd_r).sum(1) - 1
    scale = np.minimum(1.0 + crowd_gain * crowd, crowd_max)
    v = pts - pts.mean(0)
    norm = np.hypot(v[:, 0], v[:, 1])
    out_dir = np.where(norm[:, None] > 1e-6, v / np.maximum(norm, 1e-6)[:, None],
                       base / reach)
    # manual swings, for names the shortest-move rule sends the wrong way
    turned = np.zeros(n, bool)
    for k, lab in enumerate(labels):
        deg = (rotate or {}).get(str(lab))
        if deg is None:
            continue
        a = np.radians(deg)
        c_, s_ = np.cos(a), np.sin(a)
        out_dir[k] = [c_ * out_dir[k, 0] - s_ * out_dir[k, 1],
                      s_ * out_dir[k, 0] + c_ * out_dir[k, 1]]
        turned[k] = True

    # a boosted name is sent outward even when its neighbourhood is not crowded,
    # which is the only way to move it when leaders are off and the rest hug
    for k, lab in enumerate(labels):
        f = (boost or {}).get(str(lab))
        if f is not None:
            scale[k] *= f
            turned[k] = True
    rest = np.where(((crowd > 0) | turned)[:, None],
                    out_dir * reach * np.maximum(scale, 2.0)[:, None],
                    np.tile(base, (n, 1)))
    off = rest.copy()                                  # pixels, from the marker

    def centres(o):
        # the annotation anchors at marker + offset and grows right and up
        return pts + o + half

    for _ in range(n_iter):
        c = centres(off)
        shift = np.zeros_like(off)

        # name against name: separate along whichever axis needs the smaller move
        for i in range(n):
            d = c[i] - c
            ox = half[i, 0] + half[:, 0] + pad - np.abs(d[:, 0])
            oy = half[i, 1] + half[:, 1] + pad - np.abs(d[:, 1])
            hit = (ox > 0) & (oy > 0)
            hit[i] = False
            if not hit.any():
                continue
            use_x = ox[hit] < oy[hit]
            sx = np.where(d[hit, 0] >= 0, 1.0, -1.0)
            sy = np.where(d[hit, 1] >= 0, 1.0, -1.0)
            shift[i, 0] += np.sum(np.where(use_x, 0.5 * ox[hit] * sx, 0.0))
            shift[i, 1] += np.sum(np.where(use_x, 0.0, 0.5 * oy[hit] * sy))

        # name against marker, including its own: a name should not sit on a dot
        for i in range(n):
            d = c[i] - pts
            ox = half[i, 0] + marker_r + pad - np.abs(d[:, 0])
            oy = half[i, 1] + marker_r + pad - np.abs(d[:, 1])
            hit = (ox > 0) & (oy > 0)
            if not hit.any():
                continue
            use_x = ox[hit] < oy[hit]
            sx = np.where(d[hit, 0] >= 0, 1.0, -1.0)
            sy = np.where(d[hit, 1] >= 0, 1.0, -1.0)
            shift[i, 0] += np.sum(np.where(use_x, ox[hit] * sx, 0.0))
            shift[i, 1] += np.sum(np.where(use_x, 0.0, oy[hit] * sy))

        # name against the fixed furniture -- one-way, since it cannot move
        for i in range(n):
            if not len(fixed):
                break
            d = c[i] - fixed[:, :2]
            ox = half[i, 0] + fixed[:, 2] + pad - np.abs(d[:, 0])
            oy = half[i, 1] + fixed[:, 3] + pad - np.abs(d[:, 1])
            hit = (ox > 0) & (oy > 0)
            if not hit.any():
                continue
            use_x = ox[hit] < oy[hit]
            sx = np.where(d[hit, 0] >= 0, 1.0, -1.0)
            sy = np.where(d[hit, 1] >= 0, 1.0, -1.0)
            shift[i, 0] += np.sum(np.where(use_x, ox[hit] * sx, 0.0))
            shift[i, 1] += np.sum(np.where(use_x, 0.0, oy[hit] * sy))

        shift += spring * (rest - off)        # keep it near its own point
        if np.abs(shift).max() < 0.05:
            break
        off += step * shift

    # A name that has travelled gets a hairline back to its point. The leader has
    # to be set at construction (an Annotation builds its arrow patch there), so
    # those labels are replaced rather than modified.
    out = []
    for k, t in enumerate(texts):
        o = off[k] / px                       # back to points
        if np.hypot(*(o - base / px)) <= leader:
            t.set_position(tuple(o))
            out.append(t)
            continue
        t.remove()
        out.append(ax.annotate(str(labels[k]), (x[k], y[k]),
                               textcoords="offset points", xytext=tuple(o),
                               fontsize=fontsize, zorder=5, gid="point_label",
                               color=cols[k],
                               arrowprops=dict(arrowstyle="-", lw=0.45,
                                               color="0.45", shrinkA=0,
                                               shrinkB=2)))
    return out


def categoricality_hist(ax, z, xlabel="categoricality (z)", bins=16,
                        headroom=1.26, ns=1.96, annotate=None, annot_fs=None):
    """Figure 1's per-region categoricality panel: one z per region.

    Grey, because a distribution of per-unit statistics is neither a null nor the
    headline contrast. The shaded band is where a region is indistinguishable from
    a single continuous cloud of its own.
    """
    z = np.asarray(z, float)
    lo = min(np.floor(z.min()) - .5, -ns - .5)
    hi = max(np.ceil(z.max()) + .5, ns + .5)
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(z, bins=edges)
    ax.hist(z, bins=edges, color=GREY)
    ax.axvline(0, color="0.45", lw=1, ls="--")
    ax.set_xlim(lo, hi)
    _, yhi = ax.get_ylim()
    # a little more room when names are written above the bars, but only a
    # little -- too much and the bars themselves become invisible slivers
    ax.set_ylim(0, yhi * (max(headroom, 1.45) if annotate else headroom))
    ax.axvspan(-ns, ns, color="0.89", lw=0, zorder=0)
    ax.text(-ns + .15, yhi * 1.12, "n.s.", ha="left", va="center", color="0.4")
    if annotate:
        # names written above the bar they fall in, stacked when they share one
        # vertical: a bin is far narrower than a name set horizontally, so
        # horizontal text from neighbouring bins runs together
        step = 0.06 * ax.get_ylim()[1]
        used = {}
        for name, v in annotate:
            b = int(np.clip(np.searchsorted(edges, v, "right") - 1, 0, bins - 1))
            k = used.get(b, 0)
            used[b] = k + 1
            ax.text((edges[b] + edges[b + 1]) / 2, counts[b] + step * (1 + 3.2 * k),
                    str(name), ha="center", va="bottom", color="black",
                    rotation=90, clip_on=False, gid="bar_label",
                    fontsize=FS_ANNOT - 2 if annot_fs is None else annot_fs)
    ax.set(xlabel=xlabel, yticks=[])
    ax.set_box_aspect(1)
    sns.despine(ax=ax, left=True)
    return ax


def null_hist(ax, obs, null, xlabel, label_null="null", label_obs="data",
              bins=30, loc="upper left", headroom=HEADROOM, p=None,
              strip_xticks=False, ns_band=False, ns=1.96):
    """Figure 1's null panel: a blue null, a firebrick line, and no y axis.

    The legend is built from proxy handles rather than from the artists, so the
    swatch is a clean square and a short line whatever the bin count -- figure 1
    does the same.

    `p` goes on a second line of the observed value's legend entry, where figure 1
    puts it. A free-floating annotation lands on the histogram as soon as the
    observed value sits near the null, which here it does.

    `strip_xticks` is for quantities whose absolute value carries no meaning (a
    silhouette depends on the number of points and the dimensionality, so only
    where the observed value falls in its OWN null is readable).
    """
    null = np.asarray(null, float)
    ax.hist(null, bins=bins, color=NULLC, alpha=0.8)
    if ns_band:
        # the same |z| < 1.96 band the categoricality panel shades, drawn here in
        # the null's own units: an observed value inside it is not distinguishable
        # from the null. On a small field the band is wide, which is the honest
        # thing to show -- it is how little a handful of points can settle.
        # named apart from the y-limit locals below, which would shadow them
        ns_lo = null.mean() - ns * null.std()
        ns_hi = null.mean() + ns * null.std()
        ax.axvspan(ns_lo, ns_hi, color="0.89", lw=0, zorder=0)
    ax.axvline(obs, color=OBS, lw=LW_OBS)
    ax.set_xlabel(xlabel)
    ax.set_yticks([])
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * headroom)
    if p is not None:
        # a permutation p is bounded below by 1/(n_draws + 1), so a small value
        # is a floor rather than a measurement: "{p:.2f}" renders 0.002 as
        # "0.00", which reads as p = 0 and claims more than the draws support.
        label_obs = (f"{label_obs}\n$p$ < 0.01" if p < 0.01
                     else f"{label_obs}\n$p$ = {p:.2f}")
    ax.legend([Patch(facecolor=NULLC), Line2D([0], [0], color=OBS, lw=LW_OBS)],
              [label_null, label_obs], frameon=False, fontsize=FS_ANNOT, loc=loc,
              handlelength=0.9, labelspacing=0.2,
              borderaxespad=0.1 if not ns_band else 1.3)
    if ns_band:
        # the band often runs past the histogram, so the label is clamped into
        # view -- placed in data coords alone it lands outside the panel
        x0, x1 = ax.get_xlim()
        ax.set_xlim(min(x0, ns_lo), max(x1, ns_hi))
        ax.text(0.03, 0.97, "n.s.", transform=ax.transAxes, ha="left", va="top",
                color="0.4")
    if strip_xticks:
        ax.set_xticks([])
    ax.set_box_aspect(1)
    sns.despine(ax=ax, left=True)
    return ax


def points(ax, x, y, c=None, size=MARKER_SIZE, **kw):
    """A figure-1 scatter: small, black-edged markers."""
    kw.setdefault("edgecolor", "black")
    kw.setdefault("linewidths", MARKER_EDGE)
    return ax.scatter(x, y, c=c, s=size, **kw)


# -------------------------------------------------------------------- the panels
def distmat(D, data, ax=None, cmap="magma", fontsize=None, cb_label=None,
            order="hierarchy", axis_note=None, note_y=-0.30):
    """The K x K matrix.

    `order` is "hierarchy" (sensory to associative, the order the regions already
    come in), "cluster" (Ward linkage with optimal leaf ordering), or an explicit
    index array. Ordering by hierarchy lets the panel be read against the anatomy
    directly, at the cost of not showing off the block structure; Ward finds the
    blocks but puts them in an order that means nothing outside this matrix.

    `axis_note` draws an arrow under the matrix saying what the ordering IS -- the
    rows and columns are sorted by something, and a matrix that does not say by
    what is just a texture.

    `magma` rather than `magma_r`: figure 1's heatmaps run dark-low to light-high,
    and the two figures are read side by side.
    """
    fontsize = FS_ANNOT - 4.5 if fontsize is None else fontsize
    ax = ax or plt.subplots(figsize=figsize())[1]
    if isinstance(order, str):
        order = (np.argsort(data.hierarchy()) if order == "hierarchy"
                 else cluster_order(D)[0])
    order = np.asarray(order)
    off = D[np.triu_indices(len(D), 1)]
    # the diagonal is zero by definition and carries no information; left in, it
    # clips to the darkest colour and becomes the most salient thing in the panel
    M = D[order][:, order].astype(float).copy()
    np.fill_diagonal(M, np.nan)
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad("white")
    im = ax.imshow(M, cmap=cmap, clim=(off.min(), off.max()))
    c = colors(data)
    ax.set_xticks(range(len(D)))
    ax.set_yticks(range(len(D)))
    ax.set_xticklabels(data.regions[order], rotation=90, fontsize=fontsize)
    ax.set_yticklabels(data.regions[order], fontsize=fontsize)
    for t, i in zip(ax.get_xticklabels(), order):
        t.set_color(c[i])
    for t, i in zip(ax.get_yticklabels(), order):
        t.set_color(c[i])
    ax.tick_params(length=1.5, width=.5, pad=1)
    ax._tick_fontsize = fontsize       # survives the `typeset` pass
    if cb_label:      # off by default: the panel reads as a pattern, not a scale,
                      # and the bar costs the matrix a quarter of its column
        cb = plt.colorbar(im, ax=ax, shrink=.62, fraction=.040, pad=.02)
        cb.set_label(cb_label, fontsize=FS_LABEL)
        cb.set_ticks([])
        cb.outline.set_linewidth(.5)
    if axis_note:
        # under the tick labels, in axes fractions, so it spans the matrix exactly
        ax.annotate("", xy=(1, note_y), xytext=(0, note_y),
                    xycoords="axes fraction", textcoords="axes fraction",
                    annotation_clip=False, zorder=6,
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color="0.25",
                                    shrinkA=0, shrinkB=0, mutation_scale=7))
        ax.text(0.5, note_y - 0.06, axis_note, transform=ax.transAxes,
                ha="center", va="top", color="0.25", fontsize=FS_LABEL)
    ax.set_box_aspect(1)
    return ax


def mds(xy, var, data, ax=None, by="module", size=MARKER_SIZE, fontsize=None,
        label=True, axis_key=True, cbar=True, cbar_kw=None, square=True,
        rotate=None):
    """Scatter of the embedded regions, coloured by module or by hierarchy."""
    fontsize = FS_ANNOT - 3 if fontsize is None else fontsize
    ax = ax or plt.subplots(figsize=figsize())[1]
    x, y = xy.T
    if by == "module":
        for m in dict.fromkeys(data.modules()):
            s = data.modules() == m
            points(ax, x[s], y[s], color=PALETTE[m], size=size, label=m)
        ax.legend(fontsize=FS_ANNOT, frameon=False, handlelength=0.9,
                  labelspacing=0.25)
    else:
        sc = points(ax, x, y, c=data.hierarchy(), cmap="viridis", size=size)
        if cbar:
            # horizontal, under the panel, and with no ticks: the hierarchy index
            # is an ordering rather than a measured quantity, so the numbers on it
            # say less than the direction does
            kw = dict(orientation="horizontal", location="bottom",
                      fraction=.055, pad=.06, shrink=.55, aspect=28)
            kw.update(cbar_kw or {})
            cb = plt.colorbar(sc, ax=ax, **kw)
            cb.set_ticks([])
            cb.outline.set_linewidth(.5)
            cb.set_label("position in cortical hierarchy", fontsize=FS_LABEL,
                         labelpad=2)
    # the embedding axes carry no units, so figure 1's corner key replaces the
    # two axis labels; the variance each axis carries goes in the arm label
    ax.set(xticks=[], yticks=[])
    if axis_key:
        # bare "PC 1"/"PC 2", as in figure 1. The share of the variance is the
        # kind of thing that belongs in the caption: spelled out on the arms it
        # doubles the width of the key and crowds a small square panel.
        pc_arrows(ax)
    else:
        ax.set_xlabel(f"region PC 1 ({var[0]:.0%})")
        ax.set_ylabel(f"region PC 2 ({var[1]:.0%})")
    if square:
        ax.margins(.13)          # room for the offset labels near the edges
        ax.set_box_aspect(1)
    else:
        # A wide panel, with PC 1 spanning its full width. Equal data aspect
        # would instead pad the x RANGE and leave the cloud sitting in the
        # middle, so the aspect here is NOT metric -- PC 2 is stretched relative
        # to PC 1. The share of the variance each axis carries is on its arm of
        # the key, which is what the axes are read from.
        ax.margins(x=.04, y=.10)
    sns.despine(ax=ax, left=True, bottom=True)
    if label:
        # LAST, and only after a draw: `label_points` solves the collisions in
        # DISPLAY space, so it has to run once the panel's geometry is final.
        # Margins, `set_box_aspect` and the colourbar all resize the axes, and
        # labels laid out before them come apart again afterwards.
        # Coloured by module -- the same mapping the matrix's tick labels use, so
        # a region is the same colour in both panels.
        ax.figure.canvas.draw()
        label_points(ax, x, y, data.regions, fontsize=fontsize,
                     offset=(.5 * np.sqrt(size), .4 * np.sqrt(size)),
                     colors=colors(data), rotate=rotate)
    return ax


def regression_axis(ax, coef, xy, n_iso=5, color="0.25", label=None):
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
        ax.text(*(c + hi * u), f" {label}", fontsize=FS_ANNOT, color=color,
                ha="left", va="center", zorder=0)
    w = np.abs((xy - c) @ v).max() * 1.15
    for s in np.linspace(lo, hi, n_iso):
        m = c + s * u
        ax.plot([m[0] - w * v[0], m[0] + w * v[0]],
                [m[1] - w * v[1], m[1] + w * v[1]],
                color=color, lw=.5, ls=":", alpha=.35, zorder=0)
    return ax


def prediction(y, preds, names, ax=None, xlabel="true", ylabel="predicted",
               title=None, point_labels=None, size=MARKER_SIZE, fontsize=None,
               line=True, legend=True):
    """Predicted vs true, one colour per model (e.g. in-sample and cross-validated)."""
    fontsize = FS_ANNOT - 3 if fontsize is None else fontsize
    ax = ax or plt.subplots(figsize=figsize())[1]
    for pred, name, col in zip(preds, names, KIND_COLORS):
        points(ax, y, pred, color=col, size=size, label=name, zorder=3)
        if line:
            fit_line(ax, y, pred, color=col)
    lim = [min(np.min(y), *[p.min() for p in preds]),
           max(np.max(y), *[p.max() for p in preds])]
    ax.plot(lim, lim, ls="--", lw=1, color="0.45", zorder=0)
    if point_labels is not None:
        off = (.5 * np.sqrt(size), .4 * np.sqrt(size))
        for _y, _p, t in zip(y, preds[-1], point_labels):
            ax.annotate(t, (_y, _p), textcoords="offset points", xytext=off,
                        fontsize=fontsize)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(fontsize=FS_ANNOT, frameon=False, handlelength=0.9,
                  labelspacing=0.25, borderaxespad=0.1)
    if title:
        ax.set_title(title)
    ax.margins(.10)              # room for the offset labels near the edges
    ax.set_box_aspect(1)
    sns.despine(ax=ax)
    return ax


def scatter(x, y, xlabel, ylabel, ax=None, identity=False, title=None,
            size=POINT_SIZE, fontsize=None, line=True, color=GREY):
    """A dense cloud -- one point per PAIR of regions, so the markers are small.

    Grey, black-edged: the pairs are not a null and not the headline result, and
    figure 1 reserves its two hues for that contrast.
    """
    ax = ax or plt.subplots(figsize=figsize())[1]
    points(ax, x, y, color=color, size=size, linewidths=POINT_EDGE, alpha=.9,
           zorder=2)
    if line:
        fit_line(ax, x, y)
    if identity:
        lim = [min(np.min(x), np.min(y)), max(np.max(x), np.max(y))]
        ax.plot(lim, lim, ls="--", lw=1, color="0.45", zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.set_box_aspect(1)
    sns.despine(ax=ax)
    return ax


def coefficients(ax, beta, se, names, pvals, alpha=.05):
    """The competing-variable coefficients, largest first.

    Firebrick marks the coefficients that clear `alpha`, grey the rest -- the
    same "this is the result / this is the reference" contrast the histograms use,
    instead of the black-versus-grey of the earlier version.
    """
    o = np.argsort(-np.abs(beta))
    ax.bar(range(len(o)), beta[o], width=.72,
           color=[OBS if pvals[k] < alpha else GREY for k in o],
           edgecolor="black", linewidth=MARKER_EDGE)
    ax.errorbar(range(len(o)), beta[o], yerr=np.asarray(se)[o], fmt="none",
                ecolor="0.25", elinewidth=.8, capsize=1.5)
    ax.axhline(0, color="0.25", lw=.8)
    ax.set_xticks(range(len(o)))
    ax.set_xticklabels([names[k] for k in o], rotation=60, ha="right")
    ax.set_ylabel("standardised coefficient")
    ax.set_box_aspect(1)
    sns.despine(ax=ax)
    return ax


# ---------------------------------------------------------------------------
# The earlier, thinner house style (grey null, dark red data), kept because the
# head-direction and IBL figures were drawn with it. Renamed to sit beside the
# richer vocabulary above rather than collide with it: `style` -> `axis_style`,
# `null_panel` -> `simple_null_panel`, `save` -> `save_stem`.
#
# NOTE the palettes genuinely disagree: NULL_GREY below is "0.75" while NULLC
# above is a blue. Same role, different colour. Nothing is silently unified --
# a figure keeps whichever it was drawn with.
NULL_GREY, DATA_RED = "0.75", "darkred"

def axis_style(ax, xlabel, loc="upper left", headroom=1.22, fontsize=7, legend=True):
    """House style, plus headroom so the legend clears the data."""
    ax.set_xlabel(xlabel)
    ax.set_yticks([])
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * headroom)
    if legend:
        ax.legend(frameon=False, fontsize=fontsize, loc=loc, handlelength=1.0)
    ax.set_box_aspect(1)          # square plot area, so a row stays compact
    sns.despine(ax=ax, left=True)


def simple_null_panel(ax, obs, null, xlabel, label_null, label_obs, title=None,
               z=None, p=None, bins=40, loc="upper left", headroom=1.35,
               fontsize=5.5, p_fmt=".4f", annot_y=.70,
               xlabel_size="auto", tick_size="auto"):
    """One histogram of a null with the observed value as a line through it.

    `xlabel_size` / `tick_size` default to a little above the legend size; pass
    None to leave matplotlib's defaults alone, which is what the single-panel
    figures do.
    """
    ax.hist(null, bins=bins, color=NULLC, label=label_null)
    ax.axvline(obs, color=DATAC, lw=2, label=label_obs)
    if xlabel_size == "auto":
        xlabel_size = fontsize + 1.5
    if tick_size == "auto":
        tick_size = fontsize + 0.5
    ax.set_xlabel(xlabel, fontsize=xlabel_size)
    if title:
        ax.set_title(title, fontsize=8)
    if tick_size is not None:
        ax.tick_params(axis="x", labelsize=tick_size)
    axis_style(ax, xlabel, loc=loc, headroom=headroom, fontsize=fontsize)
    if z is not None:
        ax.annotate(f"z = {z:+.2f}, p = {p:{p_fmt}}", (.04, annot_y),
                    xycoords="axes fraction", fontsize=fontsize, va="top")


def save_stem(fig, stem, formats=("svg", "pdf", "png"), dpi=300):
    """Write one figure in every format the paper needs."""
    fig.tight_layout()
    for e in formats:
        fig.savefig(f"{stem}.{e}", bbox_inches="tight",
                    dpi=dpi if e == "png" else None)
