"""Per-neuron tuning curves for the IBL Repeated-Site sessions behind the main figure.

The autism analysis (`extract_asd_kernels.py`) works from released GAM kernels; this
release has no kernels, so the comparable quantity is computed directly from spikes:
each neuron's trial-averaged response over 9 signed contrasts x 10 time bins, giving a
90-dimensional vector.

Every choice here matches `ibl_analyses/scripts/setup.py`, the pipeline behind the figure
in `combined_figure.ipynb` -- response-aligned, 0 to 400 ms, 10 bins, probe00, the five
Repeated-Site areas, 50-50 blocks excluded -- so the neurons are the same population that
figure summarises, just kept one by one instead of pooled into a session geometry.

The ONE client runs in local mode against the cache already on disk, so nothing is
downloaded.

    python extract_bwm_tuning.py            # writes data_bwm/bwm_tuning.npz
"""

from pathlib import Path
import sys
import traceback

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "ibl_analyses/scripts"))

import brainbox.singlecell                                    # noqa: E402
from brainbox.io.one import SpikeSortingLoader                # noqa: E402
from iblatlas.atlas import AllenAtlas                         # noqa: E402
from iblatlas.regions import BrainRegions                     # noqa: E402
from one.api import ONE                                       # noqa: E402

from setup import good_eids, params as SETUP                  # noqa: E402

CACHE = REPO / "ibl_analyses/data" / SETUP["tag"]
OUT = Path(__file__).resolve().parent / "data_bwm"
CONTRASTS = [-100.0, -25.0, -12.5, -6.25, 0.0, 6.25, 12.5, 25.0, 100.0]


def signed_contrast(trials):
    return np.diff(np.nan_to_num(
        np.c_[trials["contrastLeft"], trials["contrastRight"]])) * 100


def session_curves(one, eid, atlas):
    """(n_neurons, 90) tuning curves and their Beryl acronyms, or None."""
    trials = one.load_object(eid, "trials", collection="alf")
    sl = SpikeSortingLoader(eid=eid, pname=SETUP["probe"], one=one, atlas=atlas)
    spikes, clusters, channels = sl.load_spike_sorting()
    if spikes is None or len(spikes) == 0:
        return None
    clusters = sl.merge_clusters(spikes, clusters, channels)

    beryl = BrainRegions().acronym2acronym(clusters["acronym"], "Beryl")
    keep = np.isin(beryl, SETUP["areas"])
    if keep.sum() == 0:
        return None

    bin_size = (SETUP["post_time"] + SETUP["pre_time"]) / SETUP["n_bins"]
    align = trials["response_times"] if SETUP["align_to"] == "response" \
        else trials["stimOn_times"]

    y, _ = brainbox.singlecell.bin_spikes2D(
        spike_times=spikes.times, spike_clusters=spikes.clusters,
        cluster_ids=clusters["cluster_id"], align_times=align,
        pre_time=SETUP["pre_time"], post_time=SETUP["post_time"], bin_size=bin_size)
    y = y[:, keep, :]

    contrast = signed_contrast(trials)
    valid = (np.isfinite(align) & np.isfinite(contrast).ravel()
             & (trials["probabilityLeft"] != 0.5))       # 50-50 blocks excluded
    y, contrast = y[valid], contrast[valid]

    levels, index, counts = np.unique(contrast, axis=0, return_inverse=True,
                                      return_counts=True)
    if not np.allclose(np.sort(levels.ravel()), CONTRASTS):
        return None                                      # a contrast is missing

    n_per = counts.min()                                 # balance the conditions
    curves = np.stack([y[np.where(index == i)[0][:n_per]].mean(0)
                       for i in range(len(levels))])     # (contrast, neuron, bin)
    n_c, n_u, n_b = curves.shape
    curves = curves.transpose(1, 0, 2).reshape(n_u, n_c * n_b)   # contrast-major
    return curves.astype(np.float32), np.asarray(beryl)[keep], int(n_per)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    one = ONE(cache_dir=CACHE, mode="local")
    atlas = AllenAtlas()

    rows = []
    for i, eid in enumerate(good_eids, 1):
        try:
            got = session_curves(one, eid, atlas)
        except Exception:
            print(f"[{i}/{len(good_eids)}] {eid[:8]} FAILED", flush=True)
            traceback.print_exc(limit=1)
            continue
        if got is None:
            print(f"[{i}/{len(good_eids)}] {eid[:8]} unusable", flush=True)
            continue
        curves, acr, n_per = got
        try:
            subject = one.get_details(eid)["subject"]
        except Exception:
            subject = eid[:8]
        rows.append(dict(curves=curves, region=acr, subject=subject, eid=eid))
        print(f"[{i}/{len(good_eids)}] {subject:<12} {curves.shape[0]:>4} neurons  "
              f"{n_per:>3} trials/cond", flush=True)

    if not rows:
        sys.exit("no sessions read -- check the cache path")

    X = np.concatenate([r["curves"] for r in rows])
    out = OUT / "bwm_tuning.npz"
    np.savez_compressed(
        out, X=X,
        region=np.concatenate([r["region"] for r in rows]),
        subject=np.concatenate([[r["subject"]] * len(r["curves"]) for r in rows]),
        eid=np.concatenate([[r["eid"]] * len(r["curves"]) for r in rows]),
        contrasts=np.array(CONTRASTS), n_bins=SETUP["n_bins"])
    subj = np.concatenate([[r["subject"]] * len(r["curves"]) for r in rows])
    print(f"\nwrote {out}")
    print(f"  {X.shape[0]} neurons x {X.shape[1]} features "
          f"({len(CONTRASTS)} contrasts x {SETUP['n_bins']} bins)")
    print(f"  {len(np.unique(subj))} subjects, {len(rows)} sessions")


if __name__ == "__main__":
    main()
