"""Small helpers for the workshop notebook.

Nothing here is part of the shape-metrics analysis itself, its just processing and plotting fuinctions.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def preprocess(X, n_pcs):
    """Conditions x neurons -> conditions x n_pcs, unit Frobenius norm."""
    X = PCA(n_pcs).fit_transform(X)      # also centers the columns
    return X / np.linalg.norm(X)


def hierarchy_colors(data, cmap="viridis"):
    """One color per region, by its position in the cortical hierarchy."""
    h = data.hierarchy().astype(float)
    norm = (h - h.min()) / (h.max() - h.min())
    return plt.get_cmap(cmap)(norm)


def label_points(ax, x, y, labels, color="0.2", fontsize=7, dx=3, dy=3):
    """A small text label next to each point (no collision avoidance --
    fine for the handful of regions used here)."""
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), xytext=(dx, dy), textcoords="offset points",
                    fontsize=fontsize, color=color)


def clean_axes(ax):
    """Drop the top/right spines -- the house style used throughout."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
