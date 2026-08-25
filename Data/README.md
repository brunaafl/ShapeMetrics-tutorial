# Data

Named by the paper the data comes from, matching the citation keys in the
manuscript's `refs.bib`.

## Two tiers

The analysis tree as a whole is 67 GB, so `Data/` is a **namespace with a
resolver**, not a container. What lives here depends on what a clone needs.

| tier | where | in git | size |
|---|---|---|---|
| **derived** | `Data/derived/<paper>/` | **yes** | 177 MB total |
| **raw** | wherever `data_roots.toml` points | no | 67 GB; one file is 29.8 GB |

**The rule: no notebook reads outside `Data/derived/` and its own `results/`.**
Anything reaching for raw data is an extraction script in disguise and belongs in
`extract/` — which runs in a different conda environment and writes back here.

That rule is what lets someone clone this repository and reproduce every figure
and every panel without downloading anything. It is now true; it used to be an
aspiration. There was a third tier in between — derived files too large for git,
listed in `MANIFEST.toml` and found through `data_roots.toml` — and four
notebooks read it, so those four could not run from a clone. The largest of them
is now 64 MB in git, which is the price of the rule holding.

The four exceptions are the Figure 3 panel notebooks `canonical_circuit`,
`inhibitory_analyses`, `inter_animal_variability` and `rotation_Cue`. They read
the raw `.mat` sessions directly — including `Dataset_3` and the `CellTypes.mat`
files, which no derived file covers — and they always will. They are exploratory
analyses that do not feed a compiled figure. They go through `paths.external()`,
so on a machine without the raw tier they say which dataset is missing and where
to point, rather than failing later on an empty session list.

## Datasets

| id | source paper | used by | what it is |
|---|---|---|---|
| `posani2026` | Posani et al. 2026, *Nature* | Fig 2 | RRR encoding-model coefficients over 8 task variables, IBL brain-wide map |
| `angelaki2025brain` | IBL / Angelaki et al. 2025, *Nature* | Fig 2 | the brain-wide map recordings the above is fit to |
| `harris2019` | Harris et al. 2019, *Nature* | Fig 2 | Allen mouse cortico-cortical connectivity matrix |
| `siegel2015` | Siegel, Buschman & Miller 2015, *Science* | Fig 2 | two monkeys, 7 cortical areas, context-dependent decisions |
| `duszkiewicz2024` | Duszkiewicz et al. 2024, *Nat. Neurosci.* | Fig 3 | mouse postsubiculum head-direction cells, 31 sessions |
| `noel2025` | Noel et al. 2025, *Nat. Neurosci.* | Fig 4 | autism mouse lines (Fmr1, Cntnap2, Shank3) + wild type |
| `iblreproducibility` | IBL 2024, bioRxiv | Fig 4 | repeated-site Neuropixels, 56 mice, + behaviour |
| `Siegle2021-hs` | Siegle et al. 2021, *Nature* | — | Allen Brain Observatory. Described in Methods; feeds no compiled figure. |

## Finding data

Roots resolve in order: `$SHAPEMETRICS_DATA`, then `data_roots.toml`
(gitignored — copy `data_roots.example.toml`), then `Data/external/`.

```python
from shapemetrics import paths
paths.derived("iblreproducibility", "bwm_tuning.npz")   # tracked, always here
paths.external("duszkiewicz2024", "Dataset_1")          # may be absent
```

`paths.external()` raises `MissingDataset` naming the paper, the expected path,
and the `extract/` script that regenerates it — rather than a bare
`FileNotFoundError` pointing at a stranger's home directory, which is what the
previous hardcoded paths produced.

There is a third accessor, for cached intermediates that a notebook can either
load or recompute:

```python
paths.cache("siegel2015", "region_categoricality.npz")
```

It returns this figure's `results/` copy if one is there, the tracked copy
otherwise, and — if neither exists — the `results/` path, so the notebook computes
into `results/` and never writes into `Data/derived/`. Panel notebooks are all
written as *load the cache, else compute it*, and before this the `else` branch
was the one a clone always took: the branch that needs the data a clone does not
have.

## Regenerating the derived files

`MANIFEST.toml` records each one's size, sha256, what it contains, the script or
notebook that produces it, and the conda environment that needs. Verify a copy
with:

```bash
shasum -a 256 <file>
```

Re-extraction needs raw data and the right environment; it is an occasional,
documented step, not part of running a figure. Two of the checks are automated:

```bash
python tools/verify_posani_slim.py   # the 14 MB Figure 2 file == the 178 MB one
python tools/check_standalone.py     # nothing outside Data/derived is reachable
```

## A note on Dropbox

The raw data currently sits inside a Dropbox folder. Symlinking it into
`Data/external/` is fragile — smart-sync can materialise or break symlinks. Prefer
moving raw data outside Dropbox and pointing `data_roots.toml` at it.
