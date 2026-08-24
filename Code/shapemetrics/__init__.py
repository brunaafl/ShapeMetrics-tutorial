"""Shared code for the shape-metrics paper.

One implementation of each thing the analyses have in common: the Procrustes
distance between two populations, the silhouette-versus-null clustering test,
the embeddings both are read in, the on-disk cache, and the house figure style.

    pip install -e refactored-code --no-deps      # once, per conda env
    import shapemetrics as sm

Every notebook that used to carry its own copy of `silhouette_sweep`,
`gaussian_null`, `subject_distances`, `embed`, `run` and `style` now calls these.
Defaults are the values those copies used, so results and figures are unchanged.
"""
from . import (cache, clustering, decoding, embedding, metrics,   # noqa: F401
               paths, plotting, shape)
from .cache import cached_npz                                        # noqa: F401
from .paths import (MissingDataset, derived, external,               # noqa: F401
                    require_env, results, set_figure)
from .decoding import (decodability, decoding_matrix,                # noqa: F401
                       pair_accuracy)
from .clustering import (best_silhouette, best_silhouette_and_k,     # noqa: F401
                         capped_silhouette, clustering_space, condition_space_test,
                         curve_gaussian_null,
                         gaussian_null, null_stats, pipeline_silhouette,
                         silhouette_sweep)
from .embedding import classical_mds, mds                            # noqa: F401
from .metrics import dsd, ssd                                        # noqa: F401
# Two house styles now live in one module, deliberately not merged: the paper's
# richer vocabulary (typeset/distmat/null_hist/...) and the earlier thin one used
# by the head-direction and IBL figures. They disagree on colour -- NULLC is a
# blue, NULL_GREY is "0.75" -- so the thin names are exported under their own
# spelling rather than silently taking over. A figure keeps what it was drawn with.
from .plotting import (DATA_RED, KIND_COLORS, NULLC, NULL_GREY,   # noqa: F401
                       PANEL, axis_style, module_palette, null_hist,
                       save, save_stem, simple_null_panel, typeset)
from .shape import (distance_matrix, pair_distance, pairwise,        # noqa: F401
                    preprocess, procrustes_distance)
