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

### Second change: `dr_algo.py` PCA solver

`PCA()` and `PCA(n_components=ncomp)` were called without `random_state` or
`svd_solver`. At the shapes this pipeline sees, sklearn's `svd_solver="auto"`
selects the **randomized** solver, whose `random_state` defaults to `None` — so
the dimensionality reduction depended on the global numpy RNG and the
categoricality z it feeds changed between runs.

Measured in the *unmodified* code, same input, two processes:

    region 0    5.2057314727   vs   5.2050811612
    region 1   -0.1459566106   vs   -0.1427143441
    region 2    0.8188577719   vs    0.8189193935

Both calls now pass `svd_solver="full"`: exact, deterministic, and cheap at
these sizes. Everything else — KMeans `random_state=42`, TSNE and MDS seeds — was
already seeded, so this was the only unseeded step.

This is a deviation from upstream and is recorded here deliberately. It changes
no method, only makes the existing one repeatable.
