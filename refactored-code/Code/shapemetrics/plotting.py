"""The house figure style: grey is always the null, dark red always the data."""
import seaborn as sns

# 2.1 inch panels, no left spine, no y ticks (cf. inter_animal_variability.ipynb)
PANEL, NULLC, DATAC = 2.1, "0.75", "darkred"


def style(ax, xlabel, loc="upper left", headroom=1.22, fontsize=7, legend=True):
    """House style, plus headroom so the legend clears the data."""
    ax.set_xlabel(xlabel)
    ax.set_yticks([])
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * headroom)
    if legend:
        ax.legend(frameon=False, fontsize=fontsize, loc=loc, handlelength=1.0)
    ax.set_box_aspect(1)          # square plot area, so a row stays compact
    sns.despine(ax=ax, left=True)


def null_panel(ax, obs, null, xlabel, label_null, label_obs, title=None,
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
    style(ax, xlabel, loc=loc, headroom=headroom, fontsize=fontsize)
    if z is not None:
        ax.annotate(f"z = {z:+.2f}, p = {p:{p_fmt}}", (.04, annot_y),
                    xycoords="axes fraction", fontsize=fontsize, va="top")


def save(fig, stem, formats=("svg", "pdf", "png"), dpi=300):
    """Write one figure in every format the paper needs."""
    fig.tight_layout()
    for e in formats:
        fig.savefig(f"{stem}.{e}", bbox_inches="tight",
                    dpi=dpi if e == "png" else None)
