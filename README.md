# Analysis code for *Neural population codes are structured across scales*

One folder per figure, one shared library, one place data is found.

## Setup

```bash
conda env create -f envs/environment.yml        # the analysis environment
conda activate shapemetrics
pip install -e . --no-deps                      # makes `import shapemetrics` work
```

That is the whole setup. Every notebook here runs from a clone with no data to
download and nothing else to configure: `Data/derived/` is tracked and holds
everything the notebooks read. `data_roots.toml` is only for `extract/` and for
the four raw-data notebooks listed below.

`--no-deps` is not optional. The three environments in `envs/` conflict on
purpose — `environment-allen.yml` pins `numpy<1.24` for `allensdk`, and
`environment-ibl.yml` pulls ~70 packages for `ibllib` — so pip must not
re-resolve them. Install the package into each environment you use.

One package is deliberately left out of `environment.yml` and needed only by
`Figure3/panels/inhibitory_analyses.ipynb`:

```bash
pip install --no-deps mvlearn                   # --no-deps is required; see envs/environment.yml
```

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

Each figure folder also has a `panels/` directory: the analyses behind the
figure, one question each, kept because the summary notebook prints numbers that
came from them. All of them run from a clone except four, which read the raw
`.mat` sessions directly and are exploratory rather than load-bearing:

| notebook | needs |
|---|---|
| `Figure3/panels/canonical_circuit.ipynb` | `duszkiewicz2024` Dataset_1, Dataset_2 |
| `Figure3/panels/inhibitory_analyses.ipynb` | the same, plus `CellTypes.mat` per session |
| `Figure3/panels/inter_animal_variability.ipynb` | the same |
| `Figure3/panels/rotation_Cue.ipynb` | Dataset_2, Dataset_3, `TuningCurvesCue.mat` |

Point `data_roots.toml` at a copy of the raw tree to run those; without it they
stop on the first data cell and name the dataset they wanted.

## Layout

```
Code/shapemetrics/   the shared library; `import shapemetrics as sm`
  paths.py           the only module that knows where data is
  shape.py           Procrustes distances
  clustering.py      silhouette vs Gaussian null ("categoricality")
  netrep_helpers.py  stochastic-metric helpers; used by figures 2 and 3
  vendor/posani/     the Posani et al. 2026 pipeline, vendored
Data/derived/        tracked, 177 MB; everything every notebook reads
extract/             scripts that read RAW data, in a DIFFERENT conda env
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
python tools/make_golden.py       # once, BEFORE changing anything
python tools/verify.py            # after: compares caches and rendered panels
python tools/check_standalone.py  # does a clone still reproduce this?
```

`verify.py` compares numbers first (every array in every cached `.npz`) and
pixels second. Raw PDF hashes are useless here — matplotlib stamps
`/CreationDate`, so two runs of identical code differ — which is why panels are
rendered to PNG before comparison. Differences that are understood are recorded
in `tools/known_differences.toml`, with the reason; they are printed, never
silenced.

`check_standalone.py` guards the promise at the top of this file. It parses every
notebook and asserts that nothing outside `Data/derived/` is reachable and that
every tracked file a notebook names is actually committed — the two ways a clone
quietly stops working on a machine that has the raw data sitting next to it.
