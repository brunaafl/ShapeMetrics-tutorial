# Extraction

Scripts that read **raw** data and write into `Data/`. They live here rather than
under a figure because they belong to a *conda environment*, not to a figure, and
several feed more than one.

| script | environment | reads | writes |
|---|---|---|---|
| `ibl_bwm_tuning.py` | `shapemetrics-ibl` | IBL repeated-site ONE cache | `Data/derived/iblreproducibility/bwm_tuning.npz` |
| `ibl_bwm_behavior.py` | `shapemetrics-ibl` | same | `.../bwm_behavior.npz` |
| `ibl_asd_kernels.py` | `shapemetrics-ibl` | noel2025 release | `Data/.../noel2025/asd_kernels.npz` |

Each calls `paths.require_env(...)` on entry, so running one in the wrong
environment fails immediately with the right `conda activate` line instead of an
ImportError from deep inside a dependency.

The environments conflict deliberately: `allensdk` pins `numpy<1.24` and `ibllib`
pulls ~70 packages, which is why they are separate and why the package must be
installed with `pip install -e . --no-deps` in each.

Running these is an occasional, documented step. A figure notebook never needs
them — it reads the tracked `Data/derived/` tier.
