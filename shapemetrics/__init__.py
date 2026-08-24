"""Shared code for the shape-metrics paper.

One implementation of each thing the analyses have in common: the Procrustes
distance between two populations, the silhouette-versus-null clustering test,
the embeddings both are read in, the on-disk cache, and the house figure style.

    import sys; sys.path.insert(0, str(REPO))
    import shapemetrics as sm

Every notebook that used to carry its own copy of `silhouette_sweep`,
`gaussian_null`, `subject_distances`, `embed`, `run` and `style` now calls these.
Defaults are the values those copies used, so results and figures are unchanged.
"""
from . import (cache, clustering, decoding, embedding, metrics,   # noqa: F401
               plotting, shape)
from .cache import cached_npz                                        # noqa: F401
from .decoding import (decodability, decoding_matrix,                # noqa: F401
                       pair_accuracy)
from .clustering import (best_silhouette, best_silhouette_and_k,     # noqa: F401
                         capped_silhouette, clustering_space, condition_space_test,
                         curve_gaussian_null,
                         gaussian_null, null_stats, pipeline_silhouette,
                         silhouette_sweep)
from .embedding import classical_mds, mds                            # noqa: F401
from .metrics import dsd, ssd                                        # noqa: F401
from .plotting import DATAC, NULLC, PANEL, null_panel, save, style   # noqa: F401
from .shape import (distance_matrix, pair_distance, pairwise,        # noqa: F401
                    preprocess, procrustes_distance)
