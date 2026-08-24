# Vendored third-party code

## `posani_utils/`

The single-area clustering pipeline of Posani et al. (2026), *Rarely categorical,
highly separable representations along the cortical hierarchy* (Nature). It computes
the "categoricality" statistic used in Figures 1, 2 and 3.

- **Upstream**: the authors' `clustering-analysis` repository, `single_area/utils/`.
- **Copied from**: `Posani/clustering-analysis/single_area/utils` in this repo, which
  was a git clone nested inside an analysis directory (its `.git`, an 18 MB pack, is
  not reproduced here).
- **Only change**: sibling imports rewritten from `from utils.X import Y` to
  `from .X import Y`, since the code is now a subpackage rather than a `sys.path` entry.
  No logic was touched.

It is vendored, not imported from its original location, because
`shapemetrics/clustering.py` previously did `sys.path.insert(...)` into that clone --
so the shared library depended on an analysis directory, and neither could be moved
or installed independently.
