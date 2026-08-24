"""Head-direction tuning curves from the Duszkiewicz et al. (2024) sessions.

The loading that every Figure 3 notebook used to repeat inline, in one place and
routed through `paths`, so it no longer depends on the working directory or on a
literal path. Three of the original notebooks pointed at `/Users/jbarbosa/...`,
a username that does not exist on this machine, so they could not run at all.

Reading the raw `.mat` sessions needs the 9.8 GB Dataset_1/Dataset_2 -- the raw
tier, absent on most machines. `tuning_curves()` therefore prefers the small
tracked cache and only falls back to the raw sessions, which is the tier rule:
a figure notebook should not need raw data.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from shapemetrics import paths

#: `hAll1`/`hAll2` are (360 bins, n_neurons, 13 smoothing levels). Index 12 is the
#: 12-degree Gaussian the Methods describe; 1 and 2 are the two cross-validated
#: halves, averaged. Sessions with fewer than MIN_NEURONS units are dropped.
SMOOTHING = 12
MIN_NEURONS = 40
CACHE = "hd_tuning.npz"


def sessions() -> list[str]:
    """Every session directory across Dataset_1 and Dataset_2, sorted.

    Raises MissingDataset, naming the paper and the size, if the raw data is not
    on this machine.
    """
    out: list[str] = []
    for ds in ("Dataset_1", "Dataset_2"):
        root = paths.external("duszkiewicz2024", ds)
        out += sorted(glob.glob(str(root / "*")))
    return [s for s in out if Path(s).is_dir()]


def session_curves(session: str):
    """(n_neurons, 360) HD tuning, averaged over the two folds. None if unusable.

    Named variables, not "the first array in the file": that heuristic picks
    hdInfoJit1 (101, 13), which is spatial-information-per-smoothing, not tuning.
    """
    f = Path(session) / "Analysis" / "HdTuning_xval_moveEp.mat"
    if not f.exists():
        return None
    hd = loadmat(str(f))
    if "hAll1" not in hd or "hAll2" not in hd:
        return None
    a, b = hd["hAll1"][:, :, SMOOTHING], hd["hAll2"][:, :, SMOOTHING]
    if a.shape[1] < MIN_NEURONS:
        return None
    return (a.T + b.T) / 2


def tuning_curves(rebuild: bool = False):
    """Tuning curves for every neuron, with the subject each came from.

    Returns (X, subject). Served from Data/derived/duszkiewicz2024/hd_tuning.npz
    unless `rebuild`, which needs the raw sessions.
    """
    if not rebuild:
        try:
            z = np.load(paths.derived("duszkiewicz2024", CACHE), allow_pickle=True)
            return z["X"], z["subject"].astype(str)
        except paths.MissingDataset:
            pass                      # fall through and build it from raw

    X, subject = [], []
    for s in sessions():
        a = session_curves(s)
        if a is None:
            continue
        X.append(a)
        subject += [Path(s).name.split("-")[0]] * a.shape[0]
    if not X:
        raise RuntimeError("no sessions read -- check data_roots.toml")
    X = np.concatenate(X)
    return X, np.asarray(subject)


def write_cache(X, subject) -> Path:
    """Save the small tracked cache, so the figure stops needing 9.8 GB of .mat."""
    out = paths.DERIVED / "duszkiewicz2024" / CACHE
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=np.asarray(X, dtype=np.float32),
                        subject=np.asarray(subject))
    return out
