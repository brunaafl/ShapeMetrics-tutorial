"""Repackage the Siegel et al. (2015) cross-validated folds into the tracked tier.

`Siegel_10_folds_avg_stim_netrep.pkl` is 76.9 MB of float64 held as a pickle:
10 folds x 2 halves x 14 unit-groups (7 areas x 2 monkeys), each a
`conditions x neurons` array. Three Figure 2 panel notebooks load it, and they
load it unconditionally -- the split-half reliability they filter on is computed
from it -- so without it they do not run at all.

Nothing here is dropped or downcast. The same float64 arrays go into a
compressed `.npz` under flat `fold|half|unit` keys, which is 64 MB: a third
smaller than the pickle, small enough to track, and not a pickle, so loading it
does not execute whatever the file happens to contain.

    conda activate shapemetrics
    python extract/siegel_folds.py

`Figure2/code/siegel_setup.folds()` reads the result and rebuilds the nested
structure, so the notebooks see exactly what `pickle.load` used to give them.
"""
from __future__ import annotations

import pickle

import numpy as np

from shapemetrics import paths

SRC = "Siegel_10_folds_avg_stim_netrep.pkl"
OUT = "siegel_folds.npz"


def main() -> None:
    src = paths.external("siegel2015", SRC)
    with open(src, "rb") as fh:
        folds = pickle.load(fh)

    flat = {}
    for i, fold in enumerate(folds):
        for h, half in enumerate(fold):
            for unit, arr in half.items():
                flat[f"{i}|{h}|{unit}"] = np.asarray(arr)      # float64, unchanged

    out = paths.DERIVED / "siegel2015" / OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **flat)
    print(f"{len(folds)} folds x {len(folds[0])} halves x {len(folds[0][0])} units")
    print(f"{src.name}  {src.stat().st_size / 1e6:.1f} MB"
          f"  ->  {out.name}  {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
