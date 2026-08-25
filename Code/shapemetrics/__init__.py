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
               paths, shape)
from .cache import cached_npz                                        # noqa: F401
# NB: `cache` is deliberately NOT re-exported here. `shapemetrics.cache` is
# already the module imported above, and `from .paths import cache` silently
# rebound that name to the function -- the same class of collision the
# palette merge made. Call it as `paths.cache(...)`, which is how the
# notebooks do; `cached_npz` from the module is exported below as before.
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
# plotting is NOT imported here, deliberately.
#
# It applies the paper's rcParams at module scope (pdf.fonttype 42 so Affinity
# gets editable text, svg.fonttype "none"), which is right for the figures drawn
# with it -- but as an eager import it fired on `import shapemetrics` and
# restyled every PDF written afterwards. Figure 1's panel silently changed:
# 40,390 -> 43,471 bytes, 20,227 pixels, from a library import it never asked for.
#
# Accessed lazily instead, so `import shapemetrics` has no global side effect
# while `from shapemetrics import plotting` still applies the house style.
_PLOTTING_NAMES = {
    "plotting", "DATA_RED", "KIND_COLORS", "NULLC", "NULL_GREY", "OBS", "PANEL",
    "PANEL_SIMPLE",
    "axis_style", "module_palette", "null_hist", "save", "save_stem",
    "simple_null_panel", "typeset",
}


#: Pre-refactor name -> what it is called now. The merge that folded the older
#: house style into this package renamed everything that collided, and the
#: notebooks were not all updated with it -- five call sites across four
#: notebooks still used an old name, and each died on a bare "has no attribute",
#: which says nothing about where the thing went. Two of them were worse than a
#: crash: `sm.save(fig, OUT / "x")` still resolves, because `save` exists -- it
#: just means something else now, and takes a bare name rather than a stem.
_RENAMED = {
    "style": "axis_style",
    "null_panel": "simple_null_panel",
    "DATAC": "DATA_RED",
    # NULLC and PANEL still exist, but as the NEW palette's blue and 1.95in --
    # a caller wanting the old grey and 2.1in gets the wrong value silently
    # rather than an error, so they are not listed: nothing here can catch that.
    # `save` is likewise still a name, with different semantics; see above.
}


def __getattr__(name):                      # PEP 562
    if name in _RENAMED:
        raise AttributeError(
            f"{__name__}.{name} was renamed to {_RENAMED[name]!r} when the two "
            f"house styles were merged into one module.\n"
            f"  If you want the older, thinner style (grey null, dark red data, "
            f"2.1in panels), it is\n"
            f"  sm.PANEL_SIMPLE / sm.NULL_GREY / sm.DATA_RED and "
            f"sm.axis_style / sm.simple_null_panel / sm.save_stem.\n"
            f"  The newer one is sm.PANEL / sm.NULLC / sm.OBS and "
            f"plotting.save / plotting.null_hist.")
    if name in _PLOTTING_NAMES:
        # import_module, not `from . import plotting`: the latter goes through
        # getattr on this package and so re-enters __getattr__, recursing forever
        import importlib
        _p = importlib.import_module(".plotting", __name__)
        globals()["plotting"] = _p          # resolve directly from here on
        return _p if name == "plotting" else getattr(_p, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from .shape import (distance_matrix, pair_distance, pairwise,        # noqa: F401
                    preprocess, procrustes_distance)
