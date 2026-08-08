# Clustering and behavioral regression require a proper metric

A cohort of simulated individuals, each with a neural geometry **and** a behavior.
Clustering the cohort and predicting behavior from neural geometry both work when
dissimilarity is the **Procrustes shape distance**, and break when it is
**linear-regression predictivity**.

## Run

```bash
python simulate_data.py         # -> simulated_data.npz  (one example cohort)
python analyses.py              # -> results.npz         (200 seeds, ~4 min)
python make_fig.py              # -> figure.pdf          (composed, 7.2 x 9.2 in)
python make_fig.py --panels     # -> also A.pdf .. E.pdf, one file per panel
```

`figure.pdf` is the assembled figure with panel letters, ready to drop straight into the
manuscript. The individual `A.pdf`–`E.pdf` are the same panels as separate transparent
PDFs, if you would rather re-compose the layout in Affinity. Both come from the same
drawing code — each panel is rendered into a matplotlib `SubFigure` — so they cannot
drift apart.

Environment: `/Users/joaobarbosa/miniforge3/envs/shapemetrics/bin/python`.
Everything here is ray-free.

## The simulation

30 individuals encode a shared circular variable θ with band-limited periodic tuning
built from harmonics 1–4. An individual's **harmonic weight profile** sets both the shape
of its neural manifold and its behavioral acuity (`log Σ_h h² w_h²`, a
Fisher-information-like quantity that depends on every harmonic weight, so predicting it
is not a disguised 1-D regression on a single generative scalar).

Three ground-truth **types** = three profiles: broad, intermediate, and sharp tuning.
Their manifolds are visibly a ring, a figure-8, and a trefoil (panel A).

Independently of type, individuals differ in how many **other** variables they encode
(0–4, "richness"). Richness is irrelevant to behavior, but it is what linear predictivity
is most sensitive to: a richer individual can linearly predict a simpler one, never the
reverse. At the chosen weight (0.10) this nuisance is invisible to Procrustes — it
correlates with Procrustes distance at r = 0.05, versus r = 0.94 for type — so the
failure below is specific to predictivity and not an artifact of a knob that hurts
everything.

## Dissimilarities

| | definition | symmetric? |
|---|---|---|
| Procrustes | `LinearMetric(alpha=1, center_columns=True, score_method='euclidean')` | yes |
| predictivity `D` | `1 − R²(i→j)`, cross-validated ridge held out over conditions | **no** |
| `Dᵀ` | the same numbers, other orientation | **no** |
| symmetrized | `(D + Dᵀ)/2` — the obvious defence | yes, but still not a metric |

Both operate on the same PCA-reduced data with the same number of components, and the
ridge penalty is set by inner cross-validation, so the comparison is not stacked.

## Results (200 seeds, mean ± sd)

| dissimilarity | ARI vs ground truth | kNN R² behavior | triangle violations | negative eigenvalue mass |
|---|---|---|---|---|
| **Procrustes** | **1.000 ± 0.000** | **+0.993 ± 0.002** | **0.0%** | **0.0%** |
| predictivity `D` | 0.460 ± 0.218 | −0.738 ± 0.449 | 9.4% | 32.8% |
| predictivity `Dᵀ` | 0.025 ± 0.065 | +0.665 ± 0.097 | 13.3% | 40.6% |
| predictivity symmetrized | 0.205 ± 0.104 | +0.386 ± 0.167 | 17.1% | 28.2% |

Three things to read off this table.

**Behavioral regression fails outright.** With the raw predictivity matrix, kNN R² is
**negative** — worse than ignoring the neural data and predicting the cohort mean. With
an asymmetric matrix "the neighbours of *i*" is not a well-defined set.

**Clustering has no single answer.** `D` and `Dᵀ` are the same measurements in two
orientations, and there is no principled reason to prefer one. They give ARI 0.460 and
0.025. Across seeds they disagree by |ΔARI| = 0.435 on average, and **disagree on 98% of
seeds**. "Which individuals form a group" is not determined by the data.

**Symmetrizing does not repair it.** `(D + Dᵀ)/2` restores symmetry and partially rescues
kNN, but it still violates the triangle inequality in **17.1%** of triples and still has
28% of its eigenvalue mass negative — so it does not embed in a Euclidean space, and the
guarantees that hierarchical clustering and nearest-neighbour regression rely on do not
hold. Procrustes: 0% and 0%.

### The asymmetry, in the extreme

Two constructed individuals, one encoding a single latent variable (participation ratio
2.8) and one encoding five (participation ratio 10.5):

```
R²(high-dim → low-dim) =  1.000
R²(low-dim → high-dim) =  0.019
Procrustes asymmetry   =  0.00e+00
```

Across the cohort, predictivity's mean |D − Dᵀ| / mean D is **1.20**; for Procrustes it is
**0.000** (max entrywise deviation 4.6e-13, i.e. floating point).

## Richness-weight sweep (30 seeds each)

How the nuisance factor trades off. Column `asym` is predictivity's asymmetry.

| weight | ARI Proc | ARI pred | kNN Proc | kNN pred | asym |
|---|---|---|---|---|---|
| 0.00 | 1.000 | 1.000 | 0.993 | −1.384 | 1.634 |
| 0.05 | 1.000 | 0.700 | 0.992 | −0.573 | 1.570 |
| **0.10** | **1.000** | **0.415** | **0.992** | **−0.622** | **1.205** |
| 0.20 | 0.773 | 0.094 | 0.984 | −0.893 | 0.721 |
| 0.35 | 0.473 | 0.019 | 0.787 | −0.617 | 0.728 |
| 0.70 | −0.003 | 0.014 | 0.458 | −0.153 | 1.211 |

Reported honestly, both directions:

- At **weight 0** — no nuisance at all — predictivity clusters perfectly (ARI 1.000), yet
  its behavioral regression is still catastrophic (R² = −1.384). The regression failure
  comes from asymmetry alone and needs no nuisance factor.
- Above **weight 0.2** the nuisance starts to contaminate Procrustes too, and by 0.7 both
  methods fail. The figure is generated at 0.10, where Procrustes is demonstrably
  untouched (r = 0.05 with richness).

## Panels

| panel | content |
|---|---|
| **a** | the cohort: tuning curves and 3-D manifolds for one individual per type, with behavior |

The individuals shown in panel a are **matched on richness** (`EXAMPLE_RICHNESS` in
`make_fig.py`, default 0). This matters: the extra encoded variables vary independently
across conditions, so an individual with high richness looks jagged when its response is
plotted against θ alone. Richness is drawn independently of type — cohort means 2.00 /
1.20 / 2.30 — so picking examples without matching can imply a link between broad tuning
and noisy tuning curves that does not exist. Set `EXAMPLE_RICHNESS = 2` (the cohort
median) to show representative individuals instead; the curves are then jagged in all
three types, which is honest but harder to read.

| **b** | asymmetry: `d(i→j)` vs `d(j→i)`; Procrustes on the identity line, predictivity far off it |
| **c** | clustering: the three dissimilarity matrices, the clusters each produces, ARI over seeds |
| **d** | behavioral regression: predicted vs true acuity, and R² over seeds |
| **e** | why: triangle violations and negative eigenvalue mass |

Panel C shows matrices rather than dendrograms deliberately: `scipy.linkage` requires a
symmetric input, and `symmetrize(D) == symmetrize(Dᵀ)`, so dendrograms would hide exactly
the order-dependence the panel exists to show.

## Notes

- `analyses.py` asserts on every seed that Procrustes is symmetric to < 1e-9 and has zero
  triangle violations. If those ever fail, the harness is wrong, not the mathematics.
- Do **not** import `ibl_analyses/scripts/utils.py` here. Its `dsd` sends every pair
  through a blocking `ray` round-trip, and its `alpha` defaults to `0.` — the whitened /
  CCA-like metric, not Procrustes.
- Nothing in this directory touches the manuscript or the existing `simple-ring/`
  simulation.
