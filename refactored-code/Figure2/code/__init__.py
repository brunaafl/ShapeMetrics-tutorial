"""Code specific to Figure 2: IBL brain-wide map (a-g) and Siegel monkey (h-k).

`shape.distance_matrix` here is NOT `shapemetrics.distance_matrix` -- this one
subsamples neurons per region and repeats; the shared one reduces to PCs and
compares once. Same name, different question. Kept apart deliberately.
"""
from . import (data, decoding, netrep_helpers, procrustes,   # noqa: F401
               shape, siegel_setup)
