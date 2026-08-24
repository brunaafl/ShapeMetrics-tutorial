# Analysis code for *Neural population codes are structured across scales*

One folder per figure, one shared library, one place data is found.

## Setup

```bash
conda env create -f envs/environment.yml        # the analysis environment
conda activate shapemetrics
pip install -e . --no-deps                      # makes `import shapemetrics` work
```

`--no-deps` is not optional. The three environments in `envs/` conflict on
purpose — `environment-allen.yml` pins `numpy<1.24` for `allensdk`, and
`environment-ibl.yml` pulls ~70 packages for `ibllib` — so pip must not
re-resolve them. Install the package into each environment you use.

## The figures

The paper compiles exactly four figures. Each has one notebook that produces
every panel it owns.

| Figure | Paper label | Notebook | Data | Source directory it replaces |
|---|---|---|---|---|
| 1 | `fig:schematic` | `Figure1/figure1.ipynb` | simulated — none | `clustering-simulation/`, `schematic/` |
| 2 | `fig:allen` * | `Figure2/figure2.ipynb` | `posani2026`, `harris2019`, `siegel2015` | `Posani/`, `siegel_analyses/` |
| 3 | `fig:hdcells` | `Figure3/figure3.ipynb` | `duszkiewicz2024` | `duszkiewicz_analyses/` |
| 4 | `fig:ibl` | `Figure4/figure4.ipynb` | `iblreproducibility`, `noel2025` | `new_IBL_analyses/` |

\* The label says `allen`, but Figure 2 is the IBL brain-wide map plus the Siegel
monkey data. The Allen Brain Observatory is described in Methods and feeds no
compiled figure. Left as-is: the manuscript is out of scope for this refactor.

The final figures are assembled by hand in Affinity Designer from these panels
(`figures/*.afdesign`). Nothing here regenerates a compiled figure — the target
is always the panel PDFs.

## Layout

```
Code/shapemetrics/   the shared library; `import shapemetrics as sm`
  paths.py           the only module that knows where data is
  shape.py           Procrustes distances
  clustering.py      silhouette vs Gaussian null ("categoricality")
  vendor/posani/     the Posani et al. 2026 pipeline, vendored
Data/derived/        small, tracked; enough to rebuild every figure
Data/external/       raw and large derived; never in git
extract/             scripts that run in a DIFFERENT conda env
Figure1..4/          notebook + code/ + panels/ + results/
tools/               golden baseline and the verification scripts
```

## Running a figure

```python
from shapemetrics import paths
import shapemetrics as sm

paths.set_figure("Figure3")                       # scopes results/ to this figure
X   = paths.derived("duszkiewicz2024", "hd_tuning.npz")
out = paths.results("hd_clustering.npz")
```

A figure notebook reads only `Data/derived/` and its own `results/`. Anything
that reaches for raw data belongs in `extract/`, which runs in a different
environment and writes back into `Data/`.

## Verifying a change

```bash
python tools/make_golden.py      # once, BEFORE changing anything
python tools/verify.py           # after: compares caches and rendered panels
```

`verify.py` compares numbers first (every array in every cached `.npz`) and
pixels second. Raw PDF hashes are useless here — matplotlib stamps
`/CreationDate`, so two runs of identical code differ — which is why panels are
rendered to PNG before comparison.
