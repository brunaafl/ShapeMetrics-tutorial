"""Code specific to Figure 2: IBL brain-wide map (a-g) and Siegel monkey (h-k).

`shape.distance_matrix` here is NOT `shapemetrics.distance_matrix` -- this one
subsamples neurons per region and repeats; the shared one reduces to PCs and
compares once. Same name, different question. Kept apart deliberately.
"""
from . import data, decoding, procrustes, shape, siegel_setup   # noqa: F401

# netrep_helpers moved to the shared library -- Figure 3's panels use it too.
# Re-exported so `paths.figure_code("Figure2").netrep_helpers` still resolves.
from shapemetrics import netrep_helpers                          # noqa: F401
