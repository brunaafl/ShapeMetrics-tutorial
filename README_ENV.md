# Environment setup

## Quick start

```bash
conda env create -f environment.yml
conda activate shapemetrics
pip install --no-deps mvlearn        # see below -- cannot live in the yml
python -m ipykernel install --user --name shapemetrics --display-name "Python (shapemetrics)"
```

That runs everything in the repo except the two data-download steps and
`gam_refit/` — see below.

A Jupyter kernel is registered as **"Python (shapemetrics)"** — pick it from the
kernel menu when opening any notebook.

If you don't have conda, install Miniforge first:

```bash
curl -L -o mf.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh
bash mf.sh -b -p "$HOME/miniforge3"
"$HOME/miniforge3/bin/conda" init zsh    # then restart your shell
```

(Swap `MacOSX-x86_64` for `MacOSX-arm64` or `Linux-x86_64` as appropriate.)

---

## Environments, and why there is more than one

| environment | for | file |
|---|---|---|
| `shapemetrics` | every analysis and figure in the repo | `environment.yml` |
| `shapemetrics-ibl` | downloading IBL data (`one`, `ibllib`, `iblatlas`) | `environment-ibl.yml` |
| `shapemetrics-allen` | downloading Allen Brain Observatory data (`allensdk`) | `environment-allen.yml` |

Plus `gam_refit/`, which is a vendored upstream clone with its own toolchain —
see `gam_refit/README.md`. It is not covered by any of the three files above.

**The split is only about the two data-access APIs.** Everything downstream of
the download — every figure script, every notebook, every metric computation —
runs in `shapemetrics`.

`allensdk` pins `numpy<1.24` and `pandas==1.5.3`; installing it into the main
environment drags numpy from 2.4 back to 1.23.5 and breaks scikit-learn 1.9.

`ibllib` is less violent but not free: it pulls ~70 packages (PyQt5,
scikit-image, dask, pandera, boto3, numba, …) and pins numpy back to 2.3.

Both are only needed to *fetch* data, so both get their own environment:

```bash
# Allen
conda activate shapemetrics-allen
python allen-brain/extract_data.py          # writes allen-brain/data/extracted_data/*.npz

# IBL
conda activate shapemetrics-ibl
python Posani/brainwide-RRR-encoding-model/step1_IBL_downloaddata.py

# then, for everything else
conda activate shapemetrics
```

Note that `allen-brain/make_fig.py` also imports `allensdk` (for
`BrainObservatoryCache` and `ReferenceSpaceCache`), so it runs in
`shapemetrics-allen` too — not, as an earlier version of this file claimed, in
the main environment.

Which files need which environment:

| environment | files |
|---|---|
| `shapemetrics-allen` | `allen-brain/extract_data.py`, `allen-brain/make_fig.py` |
| `shapemetrics-ibl` | `ibl_analyses/scripts/loader.py`, `Posani/brainwide-RRR-encoding-model/step1_IBL_downloaddata.py` |
| `shapemetrics` | everything else |

---

## Things that were not obvious

**The environment is built with pip on top of a bare conda python, on purpose.**
Installing `numba`/`statsmodels` from conda-forge into this env resolves numpy
back to 1.26 and pandas to 2.3 — conda does not see the pip-installed numpy 2.4
and happily replaces it. The pip wheels resolve cleanly against numpy 2.4 /
pandas 3.0. If you add a dependency, add it to the `pip:` block.

**Python is pinned to 3.11.** Several notebooks still `import imp`, which was
removed from the standard library in 3.12. Don't bump this without removing
those imports first (`imp.reload(x)` → `importlib.reload(x)`).

**`netrep` is pinned to commit `0186b8a`**, the revision used by the paper.
`Posani/code/procrustes.py` is checked against it to 1e-15.

**`mvlearn` is a separate `pip install --no-deps` step, not a line in the yml.**
A bare `--no-deps` inside the `pip:` block does not work — conda hands it to pip
as if it were a package name and the whole `conda env create` dies with
`ERROR: Invalid requirement: --no-deps`. So it goes in by hand afterwards.

`--no-deps` itself is required because mvlearn declares
`matplotlib<=3.3.4`, a version with no current wheel, so a normal install makes
pip compile matplotlib from source and it fails building the bundled freetype.
The cap is stale — `mvlearn` is only used for `GCCA`
(`duszkiewicz_analyses/notebooks/inhibitory_analyses.ipynb`), which doesn't
touch matplotlib. Verified working against matplotlib 3.11:

```python
from mvlearn.embed import GCCA
import numpy as np
GCCA(n_components=2).fit_transform([np.random.randn(50, 8) for _ in range(3)])
```

`pip check` will report this pin as a conflict. It is expected and harmless.

**`ray` is required, not optional.** `ibl_analyses/scripts/utils.py` wraps every
pairwise distance in a `@ray.remote` call, so `import utils` fails without it —
and the `duszkiewicz_analyses` and `fmri` notebooks both
`sys.path.append("../../ibl_analyses/scripts")` and import it.

**`gam_refit/` is out of scope.** It is a clone of
github.com/BalzaniEdoardo/PGAM, and `PGAM/GAM_library.py` hard-imports `rpy2`
and calls `importr("survey")` — it needs R inside the environment plus an
OpenMP-capable compiler for its two Cython extensions. Apple clang rejects
`-fopenmp`, so the build needs `clang_osx-arm64` + `llvm-openmp` from
conda-forge. Keep that in its own env; `gam_refit/README.md` has the details.

---

## Verify the install

```bash
conda activate shapemetrics
python - <<'EOF'
import importlib, warnings
warnings.filterwarnings("ignore")
mods = ["numpy","scipy","sklearn","matplotlib","seaborn","pandas","statsmodels","xarray",
        "netCDF4","h5netcdf","h5py","joblib","tqdm","yaml","jax","ray","netrep","mord",
        "scikits.bootstrap","mvlearn","ripser","brokenaxes","decodanda","ijson","openpyxl",
        "pypdf","imp","ipykernel"]
bad = []
for m in mods:
    try: importlib.import_module(m)
    except Exception as e: bad.append((m, e))
print(f"{len(mods)-len(bad)}/{len(mods)} OK")
for m, e in bad: print("  FAIL", m, e)
EOF
```

Expect `28/28 OK`.

Then run something real (all of these use committed data, no download needed):

```bash
python schematic/make_panel_D.py            # ~5 s
python noise-simple-ring/make_noise_fig.py  # ~20 s
python simple-ring/make_fig.py              # ~2 min
python metric-comparison/make_fig.py        # ~4 min
python Posani/analysis.py                   # ~40 s
python Posani/regression.py                 # ~150 s
```

`Posani/analysis.py` should print, for the selectivity mode, within-region
`0.351 +/- 0.052`, across-region `0.513 +/- 0.130`, and a label-shuffled null of
`0.308 +/- 0.014` (z = 15.1). `metric-comparison/make_fig.py` should print
`procrustes ARI 1.000 kNN +0.993`.

---

## Versions this was built and tested against

```
python 3.11.13   numpy 2.4.6     pandas 3.0.5
sklearn 1.9.0    scipy 1.17.1    matplotlib 3.11.1
statsmodels 0.14.6   xarray 2026.7.0   ray 2.49.2
netrep 0186b8a   jax 0.4.38      seaborn 0.13.2
```

Platform: macOS, `osx-64`.

Nothing is version-pinned except `python=3.11`, the `netrep` commit, and
`numpy<1.24` / `pandas=1.5.3` (the last two only in the Allen environment). If a
future release breaks something, pin it here rather than downgrading the whole
stack.

---

## Known non-issues

- `pip check` reports `mvlearn 0.5.0 requires matplotlib<=3.3.4` — see above.
- `ray` prints a `GOAWAY`/grpc warning on shutdown. Cosmetic.
- Notebooks using `imp` emit a `DeprecationWarning` on 3.11. Cosmetic.
- scikit-learn 1.9 emits `FutureWarning`s from `manifold.MDS` about the `init`
  and `dissimilarity` parameters changing in 1.10. Cosmetic for now; worth
  fixing before 1.10.
- `np.row_stack` is deprecated (`schematic/make_panel_E.py`). Use `np.vstack`.
- Matplotlib builds a font cache on first plot. One-off.

---

## Not an environment problem: numpy 2 semantics

`ibl_analyses/scripts/utils.py` returns `np.array(ray.get(refs))` from both
`dsd()` and `ssd()`, and every caller passes a single pair and assigns the
result straight into one cell of a distance matrix:

```python
distances[0, i, j] = utils.dsd([[sub_i[0], sub_j[1]]], alpha=0)
```

That assigns a shape-`(1,)` array to a scalar slot. numpy 1.x allowed it with a
`DeprecationWarning`; numpy 2 raises

```
ValueError: setting an array element with a sequence.
```

It hits `fmri/notebooks/bold_analyse.ipynb` and every `duszkiewicz_analyses`
notebook. Installing an older numpy is the wrong fix — pandas 3 and the Posani
analyses want numpy 2. Fix it once at the source instead, in `utils.py`:

```python
     D = ray.get(refs)
     D = np.array(D)

-    return D
+    return D.item() if D.size == 1 else D
```

in both `dsd()` and `ssd()`. Every call site in the repo passes exactly one
pair, so this changes nothing else. With that patch,
`fmri/notebooks/bold_analyse.ipynb` executes all 11 cells cleanly.

(`utils.py` is currently a Dropbox placeholder — hydrate it first, see below.)

---

## Not an environment problem: missing data files

`Posani/Separability/{Fig2i,Fig5,Fig6}.py` need `data/conditioned_trials_IC.pck`,
which is not in the clone (`data/conditioned_trials.pck` is). `Fig2ef.py` and
`Fig4.py` run. `Fig2ef.py` takes a long time by design — its multiclass decode
uses `nshuffles=25` over 20+ regions.

`Posani/brainwide-RRR-encoding-model/step{1,2,3}*.py` import a `utils/` package
that is not present in the clone, and steps 2–3 additionally need `torch`. They
are the upstream training pipeline; the analyses here start from the released
`trained_RRR_model/RRR_selectivity.json`, so none of that is installed.

---

## Not an environment problem: Dropbox placeholders

Several files in the working tree are 0 bytes with a `com.dropbox.placeholder`
xattr — they are online-only and reading them does **not** hydrate them:

```
allen-brain/extract_data.py            allen-brain/make_fig.py
ibl_analyses/scripts/*.py              ibl_analyses/notebooks/*.ipynb
duszkiewicz_analyses/notebooks/*.ipynb
```

Nothing that imports them can run until Dropbox is told to make them available
offline (right-click → "Make available offline", or turn off Smart Sync for this
folder). To list them:

```bash
find . -name '*.py' -o -name '*.ipynb' | while read f; do
  xattr "$f" 2>/dev/null | grep -q dropbox.placeholder && echo "$f"
done
```

Their dependencies are nevertheless already installed in `shapemetrics`
(`ray`, `mvlearn`, `ripser`, `brokenaxes`, `scikits.bootstrap`, `jax`), audited
from the committed versions in git.
