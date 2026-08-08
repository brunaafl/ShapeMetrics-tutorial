# new_IBL_analyses

The IBL figure of the shape-metrics paper. Run `combined_figure.ipynb` top to
bottom from this directory and it regenerates every panel. It imports no local
modules: `data_fig/` is the only thing it needs.

## Layout

```
combined_figure.ipynb    the figure and its analyses
data_fig/                precomputed inputs, tracked (11 files, 6.7 MB)
figure_ibl/              output; written by the notebook, not tracked
```

`data_fig/` is in the repository, so a fresh clone runs with no setup beyond
the environment. `figure_ibl/` is not -- the notebook creates it. The arrays
come from the upstream IBL pipeline; nothing here recomputes a distance matrix.

## The figure

`figure_ibl/IBL_combined_all.pdf` — a 3 x 3 grid:

| panel | content |
|---|---|
| **a** | reserved (blank; the dataset schematic is added downstream) |
| **b** | mean shape distance between five recorded regions, Wilcoxon across the mice recorded in all five |
| **c** | psychometric curves, 56 mice |
| **d** | pairwise distance matrices; lower triangle neural, upper triangle behavioural |
| **e** | neural against behavioural distance over all mouse pairs |
| **f** | z-scored distance correlation across cross-validation folds, per region and pooled |
| **g** | genotype: Ward dendrogram over the ASD neural distances, with the ordered matrix below |
| **h** | the same animals embedded by MDS |
| **i** | leave-one-out genotype decoding, one-vs-rest, against 10,000 label shuffles |

The last cell is the figure itself, so once the notebook has been run once you
can re-run that cell alone to redraw without repeating the permutation tests.

Two further cells produce supporting analyses: the cluster-count sweep with
per-cluster psychometrics (`IBL_clusters_behaviour`), and the joint
behaviour-space test asking whether the neural clusters separate on choice and
reaction-time summaries (`IBL_behaviour_space`).

## Inputs

All arrays are precomputed elsewhere and simply loaded here.

| file | shape | what |
|---|---|---|
| `all_dist_neural.npy` | 99 x 56 x 56 | neural shape distances, one matrix per cross-validation fold |
| `all_dist_cc.npy` | 99 x 56 x 56 | behavioural distances, same folds |
| `all_c_corrs.npy` | 99 x 2 | per fold: the distance-distance correlation and its shuffle null |
| `dist_neural_nocv.npy` | 56 x 56 | neural distances, not cross-validated |
| `dist_cc_nocv.npy` | 56 x 56 | behavioural distances, not cross-validated |
| `choices.npy` | 56 x 9 | P(right) per contrast, per mouse |
| `means_dist_subject.npy` | 17 x 5 x 5 | region-by-region distances for the mice recorded in all five regions |
| `region_info.csv` | 250 rows | per-region correlation and z-score, 50 folds x 5 regions |
| `dist_neural_asd.npy` | 37 x 37 | neural distances across the genotype cohort |
| `labels_arr.npy` | 37 | genotype of each animal: N wild type, F Fmr1, C Cntnap2, S Shank3 |
| `reaction_times.npz` | 56 arrays | per mouse, (trials x 9 contrasts); RT is right-skewed, so take the median per contrast |

## Requirements

numpy, scipy, pandas, matplotlib, seaborn, scikit-learn, jax, tqdm. The repo's
`environment.yml` covers all of them.

jax is used for exactly two `jnp.triu_indices` calls; swapping them for numpy
would drop the dependency entirely.

## Notes

Distances arrive precomputed, so the metric variant (the `alpha` of the shape
distance) is not recoverable from these files and has to be stated from the
source pipeline.

Panel d shows one fold of 99, selected as the highest-correlating one.

The genotype cohort here is 37 animals (wild type 10, Fmr1 9, Cntnap2 9,
Shank3 9). Earlier drafts of the manuscript quote 33.
