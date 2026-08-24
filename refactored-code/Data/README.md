# Data

Named by the paper the data comes from, matching the citation keys in the
manuscript's `refs.bib`.

## Three tiers

The repository as a whole is 67 GB, so `Data/` is a **namespace with a resolver**,
not a container. What lives here depends on what a fresh clone needs.

| tier | where | in git | why |
|---|---|---|---|
| **derived, small** | `Data/derived/<paper>/` | **yes** (7.9 MB total) | enough to rebuild the figures |
| **derived, large** | wherever `data_roots.toml` points | no — see `MANIFEST.toml` | 22–178 MB each |
| **raw** | wherever `data_roots.toml` points | no | 67 GB; one file is 29.8 GB |

**The rule: a figure notebook reads only the small tier plus its own `results/`.**
Anything reaching for raw data is an extraction script in disguise and belongs in
`extract/` — which runs in a different conda environment and writes back here.

That rule is what lets someone clone this repository and reproduce all four
figures without downloading anything.

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

## Regenerating the large files

`MANIFEST.toml` records each one's size, sha256, the script that produces it, and
the conda environment that script needs. Verify a copy with:

```bash
shasum -a 256 <file>
```

Re-extraction needs raw data and the right environment; it is an occasional,
documented step, not part of running a figure.

## A note on Dropbox

The raw data currently sits inside a Dropbox folder. Symlinking it into
`Data/external/` is fragile — smart-sync can materialise or break symlinks. Prefer
moving raw data outside Dropbox and pointing `data_roots.toml` at it.
