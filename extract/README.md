# Extraction

Scripts that read **raw** data and write into `Data/`. They live here rather than
under a figure because they belong to a *conda environment*, not to a figure, and
several feed more than one.

| script | environment | reads | writes |
|---|---|---|---|
| `ibl_bwm_tuning.py` | `shapemetrics-ibl` | IBL repeated-site ONE cache | `Data/derived/iblreproducibility/bwm_tuning.npz` |
| `ibl_bwm_behavior.py` | `shapemetrics-ibl` | same | `.../bwm_behavior.npz` |
| `ibl_asd_kernels.py` | `shapemetrics-ibl` | noel2025 release | `.../noel2025/asd_kernels.npz` |
| `posani_selective.py` | `shapemetrics` | `rrr_neurons.npz`, 178 MB | `.../posani2026/rrr_selective.npz`, 14 MB |
| `siegel_folds.py` | `shapemetrics` | `Siegel_10_folds_avg_stim_netrep.pkl`, 77 MB | `.../siegel2015/siegel_folds.npz`, 64 MB |

The last two do not read raw recordings; they shrink a large derived file into
one small enough to track. They exist because their inputs were the reason four
notebooks could not run from a clone. `Data/MANIFEST.toml` records what each
output contains and how it relates to its input; `tools/verify_posani_slim.py`
proves the Figure 2 one loses nothing that any analysis reads.

Each calls `paths.require_env(...)` on entry, so running one in the wrong
environment fails immediately with the right `conda activate` line instead of an
ImportError from deep inside a dependency.

The environments conflict deliberately: `allensdk` pins `numpy<1.24` and `ibllib`
pulls ~70 packages, which is why they are separate and why the package must be
installed with `pip install -e . --no-deps` in each.

Running these is an occasional, documented step. No notebook needs them — every
notebook reads the tracked `Data/derived/` tier. `tools/check_standalone.py`
enforces that, so if a new notebook starts reaching past it, the check fails
rather than the next clone.
