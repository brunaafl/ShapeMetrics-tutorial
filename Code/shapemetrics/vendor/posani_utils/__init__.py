"""Posani et al. (2026) single-area clustering pipeline, vendored.

Upstream: the `clustering-analysis` repo, subdirectory single_area/utils.
Vendored rather than sys.path-imported so the shared library does not depend
on an analysis directory. Sibling imports were rewritten `from utils.X` ->
`from .X`; nothing else was changed.
"""
