"""Slim the Posani et al. (2026) RRR coefficients down to the tracked tier.

`rrr_neurons.npz` is 178 MB: every neuron in the brain-wide map, selective or
not. Figure 2 uses a strict subset of it -- the neurons that pass the paper's
own selectivity criterion (dR2 > 0.015) and sit in one of the cortical areas of
`area_list.csv`. That is 4,617 of 59,820 neurons, and it compresses to ~13 MB,
which is small enough to track.

The subset is not an approximation. `Dataset.__init__` computes exactly this
mask and then never looks at the discarded rows, so a Dataset built on the slim
file has the same `regions`, the same `counts`, and the same `features()` as one
built on the full file -- `tools/verify_posani_slim.py` asserts it.

    conda activate shapemetrics
    python extract/posani_selective.py           # writes the tracked file

Reads the raw tier, so it needs `data_roots.toml` (or $SHAPEMETRICS_DATA) to
point at a copy of `rrr_neurons.npz`. A figure notebook never runs this: the
output is in git.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from shapemetrics import paths

DR2_THRESHOLD = 0.015                       # the paper's selective-neuron criterion
OUT = "rrr_selective.npz"


def main() -> None:
    src = paths.external("posani2026", "rrr_neurons.npz")
    z = np.load(src, allow_pickle=True)
    acronym, dR2, beta = z["acronym"], z["dR2"], z["beta"]

    area_list = pd.read_csv(
        paths.derived("posani2026", "area_list.csv"), header=None).values[:, 0]
    keep = (dR2 > DR2_THRESHOLD) & np.isin(acronym, area_list)

    out = paths.DERIVED / "posani2026" / OUT
    np.savez_compressed(
        out,
        acronym=acronym[keep],
        dR2=dR2[keep],
        beta=np.asarray(beta[keep], dtype=np.float32),
        var_list=z["var_list"],
        # so a reader can tell this is a subset, and of what
        n_neurons_total=np.int64(len(acronym)),
        dr2_threshold=np.float64(DR2_THRESHOLD),
    )
    print(f"{src.name}: {len(acronym)} neurons -> {int(keep.sum())} selective")
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
