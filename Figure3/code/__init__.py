"""Code specific to Figure 3: head-direction coding in mouse postsubiculum."""
from . import hd            # noqa: F401

# Shared with Figure 2, so it lives in the library rather than in either
# figure. Re-exported here because the panel notebooks ask for it as
# `paths.figure_code("Figure3").netrep_helpers`.
from shapemetrics import netrep_helpers                          # noqa: F401
