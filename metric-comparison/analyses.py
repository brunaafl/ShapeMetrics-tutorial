"""Run the cohort simulation over many seeds and measure what breaks.

For each seed we build the Procrustes and predictivity dissimilarities and ask:
  - how asymmetric is each one?
  - do they satisfy the triangle inequality?
  - can we recover the ground-truth types by clustering?         (ARI)
  - can we predict behavior from neural geometry?                (kNN R^2)

The predictivity matrix is evaluated three ways -- D, its transpose, and the
symmetrized average -- because with an asymmetric matrix there is no principled
reason to prefer one orientation, and the answer depends on the choice.

Run:  python analyses.py
Out:  results.npz
"""
import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, r2_score

import metrics as mx
from simulate_data import N_HARMONICS, N_THETAS, RICHNESS_WEIGHT, simulate

N_SEEDS = 200
N_NEIGHBOURS = 3
N_TYPES = 3
SWEEP = [0.0, 0.05, 0.1, 0.2, 0.35, 0.7]
VARIANTS = ["procrustes", "predictivity", "predictivity_T", "predictivity_sym"]


# ---------------------------------------------------------------- downstream
def cluster_ari(D, labels):
    """Recover the ground-truth types by hierarchical clustering."""
    pred = AgglomerativeClustering(
        n_clusters=N_TYPES, metric="precomputed", linkage="average"
    ).fit_predict(D)
    return adjusted_rand_score(labels, pred)


def knn_r2(D, behavior, k=N_NEIGHBOURS):
    """Leave-one-out kNN regression of behavior from position in the space.

    Row i of D is used as "the neighbours of i".  When D is asymmetric that is a
    choice, not a fact -- using the column instead gives a different neighbour
    set, which is the whole point.
    """
    K = len(D)
    masked = D + np.eye(K) * 1e9          # never let i be its own neighbour
    pred = np.array([behavior[np.argsort(masked[i])[:k]].mean() for i in range(K)])
    return r2_score(behavior, pred)


def run_seed(seed, richness_weight=RICHNESS_WEIGHT):
    Xs, behavior, labels, _, richness = simulate(seed=seed,
                                                 richness_weight=richness_weight)
    Ps = mx.to_pcs(Xs)

    Dp = mx.pairwise_procrustes(Ps)
    Dr = mx.pairwise_predictivity(Ps, seed=seed)
    Ds = mx.symmetrize(Dr)

    # Procrustes is a metric by construction; if these ever fail the harness,
    # not the mathematics, is wrong.
    assert np.abs(Dp - Dp.T).max() < 1e-9, "Procrustes came out asymmetric"
    assert mx.triangle_violations(Dp) == 0.0, "Procrustes violated the triangle inequality"

    out = {}
    for name, D in zip(VARIANTS, [Dp, Dr, Dr.T, Ds]):
        out[name] = dict(
            ari=cluster_ari(D, labels),
            knn=knn_r2(D, behavior),
            asym=mx.asymmetry(D),
            tri=mx.triangle_violations(D),
            negeig=mx.negative_eigenvalue_mass(D),
        )

    # what is each dissimilarity actually tracking?
    iu = np.triu_indices(len(Dp), 1)
    d_type = np.abs(labels[:, None] - labels[None, :])[iu]
    d_rich = np.abs(richness[:, None] - richness[None, :])[iu]
    out["tracking"] = dict(
        proc_type=np.corrcoef(Dp[iu], d_type)[0, 1],
        proc_rich=np.corrcoef(Dp[iu], d_rich)[0, 1],
        pred_type=np.corrcoef(Ds[iu], d_type)[0, 1],
        pred_rich=np.corrcoef(Ds[iu], d_rich)[0, 1],
    )
    return out


# ------------------------------------------------- the promised extreme pair
def extreme_pair(seed=0):
    """Two individuals for which predictivity gives R^2 ~ 1 in one direction and
    R^2 ~ 0 in the other: a low-dimensional code and a high-dimensional one."""
    rs = np.random.RandomState(seed)
    M, N, D = N_THETAS, 60, 5
    latents = rs.uniform(-np.pi, np.pi, size=(M, D))

    def pop(dims, wts):
        X = np.zeros((N, M))
        for n in range(N):
            for k, w in zip(dims, wts):
                X[n] += w * np.exp(2.0 * np.cos(latents[:, k] - rs.uniform(-np.pi, np.pi)))
        return X - X.mean(axis=1, keepdims=True)

    A = pop([0], [1.0])                              # encodes one variable
    B = pop([0, 1, 2, 3, 4], [0.5, 1, 1, 1, 1])      # encodes five

    # Keep enough components that B's (down-weighted) copy of variable 0 is not
    # truncated away -- that shared dimension is the whole point of the demo.
    Pa, Pb = mx.to_pcs([A, B], n_pcs=40)
    return dict(
        r2_high_to_low=mx.predictivity_r2(Pb, Pa),
        r2_low_to_high=mx.predictivity_r2(Pa, Pb),
        d_proc_ab=mx.d_procrustes(Pa, Pb),
        d_proc_ba=mx.d_procrustes(Pb, Pa),
        pr_low=float(np.sum(np.linalg.svd(A, compute_uv=False) ** 2) ** 2
                     / np.sum(np.linalg.svd(A, compute_uv=False) ** 4)),
        pr_high=float(np.sum(np.linalg.svd(B, compute_uv=False) ** 2) ** 2
                      / np.sum(np.linalg.svd(B, compute_uv=False) ** 4)),
    )


# ---------------------------------------------------------------- main
def collect(runs, field):
    return {v: np.array([r[v][field] for r in runs]) for v in VARIANTS}


if __name__ == "__main__":
    print(f"running {N_SEEDS} seeds at richness_weight={RICHNESS_WEIGHT} ...")
    runs = Parallel(n_jobs=-2, verbose=1)(
        delayed(run_seed)(s) for s in range(N_SEEDS))

    ari = collect(runs, "ari")
    knn = collect(runs, "knn")
    tri = collect(runs, "tri")
    neg = collect(runs, "negeig")
    asym = collect(runs, "asym")

    print(f"\n{'dissimilarity':<20} {'ARI':>16} {'kNN R2 behaviour':>20} "
          f"{'tri.viol':>10} {'neg.eig':>9} {'asym':>7}")
    for v in VARIANTS:
        print(f"{v:<20} {ari[v].mean():>7.3f} +- {ari[v].std():<6.3f} "
              f"{knn[v].mean():>10.3f} +- {knn[v].std():<6.3f} "
              f"{100*tri[v].mean():>9.1f}% {100*neg[v].mean():>8.1f}% "
              f"{asym[v].mean():>7.3f}")

    trk = {k: np.mean([r["tracking"][k] for r in runs])
           for k in runs[0]["tracking"]}
    print(f"\ntracking (mean r):  Procrustes  type {trk['proc_type']:.2f}  "
          f"richness {trk['proc_rich']:.2f}")
    print(f"                    predictivity type {trk['pred_type']:.2f}  "
          f"richness {trk['pred_rich']:.2f}")

    # order dependence, the headline clustering result
    flip = np.abs(ari["predictivity"] - ari["predictivity_T"])
    print(f"\nclustering order dependence |ARI(D) - ARI(D^T)|: "
          f"mean {flip.mean():.3f}, max {flip.max():.3f}, "
          f"disagrees on {100*(flip > 0.05).mean():.0f}% of seeds")

    ep = extreme_pair()
    print(f"\nextreme pair (participation ratio {ep['pr_low']:.1f} vs {ep['pr_high']:.1f}):")
    print(f"  R2(high -> low) = {ep['r2_high_to_low']:.3f}")
    print(f"  R2(low -> high) = {ep['r2_low_to_high']:.3f}")
    print(f"  d_Proc asymmetry = {abs(ep['d_proc_ab'] - ep['d_proc_ba']):.2e}")

    # richness sweep
    print(f"\nrichness-weight sweep (30 seeds each):")
    print(f"{'weight':>7} {'ARI proc':>10} {'ARI pred':>10} "
          f"{'kNN proc':>10} {'kNN pred':>10} {'asym pred':>10}")
    sweep = {}
    for w in SWEEP:
        rs_ = Parallel(n_jobs=-2)(delayed(run_seed)(s, w) for s in range(30))
        sweep[w] = dict(
            ari_p=np.mean([r["procrustes"]["ari"] for r in rs_]),
            ari_r=np.mean([r["predictivity"]["ari"] for r in rs_]),
            knn_p=np.mean([r["procrustes"]["knn"] for r in rs_]),
            knn_r=np.mean([r["predictivity"]["knn"] for r in rs_]),
            asym_r=np.mean([r["predictivity"]["asym"] for r in rs_]),
        )
        s = sweep[w]
        print(f"{w:>7.2f} {s['ari_p']:>10.3f} {s['ari_r']:>10.3f} "
              f"{s['knn_p']:>10.3f} {s['knn_r']:>10.3f} {s['asym_r']:>10.3f}")

    np.savez(
        "results.npz",
        variants=VARIANTS,
        ari=np.array([ari[v] for v in VARIANTS]),
        knn=np.array([knn[v] for v in VARIANTS]),
        tri=np.array([tri[v] for v in VARIANTS]),
        negeig=np.array([neg[v] for v in VARIANTS]),
        asym=np.array([asym[v] for v in VARIANTS]),
        sweep_weights=np.array(SWEEP),
        sweep=np.array([[sweep[w][k] for k in
                         ("ari_p", "ari_r", "knn_p", "knn_r", "asym_r")]
                        for w in SWEEP]),
        extreme_pair=np.array([ep[k] for k in
                               ("r2_high_to_low", "r2_low_to_high",
                                "d_proc_ab", "d_proc_ba", "pr_low", "pr_high")]),
        n_seeds=N_SEEDS, richness_weight=RICHNESS_WEIGHT,
    )
    print("\nwrote results.npz")
