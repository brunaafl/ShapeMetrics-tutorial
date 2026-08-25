"""Assert that Figure 2's tracked coefficient file is not a degraded copy.

`Data/derived/posani2026/rrr_selective.npz` is what a fresh clone gets, and it
holds 4,617 of the 59,820 neurons in `rrr_neurons.npz`. That is only safe if the
discarded rows are ones `Dataset` would have masked out anyway -- otherwise the
clone would silently draw a different Figure 2, which is exactly the failure the
figure caches were tracked to avoid.

So build a Dataset both ways and compare everything the notebook reads.

    conda activate shapemetrics
    python tools/verify_posani_slim.py

Needs the 178 MB file, so it runs on a machine that has the raw tier, not in CI.
Skips with a clear message when that file is absent.
"""
from __future__ import annotations

import sys

import numpy as np

from shapemetrics import paths

sys.path.insert(0, str(paths.ROOT))
_c = paths.figure_code("Figure2")
data = _c.data


def main() -> int:
    try:
        full = paths.external("posani2026", "rrr_neurons.npz")
    except paths.MissingDataset:
        print("skip: the 178 MB rrr_neurons.npz is not on this machine.\n"
              "      Point data_roots.toml at it to run this check.")
        return 0

    slim = paths.derived("posani2026", "rrr_selective.npz")
    a, b = data.Dataset(slim), data.Dataset(full)

    checks = [
        ("regions",      lambda: np.array_equal(a.regions, b.regions)),
        ("counts",       lambda: np.array_equal(a.counts, b.counts)),
        ("hierarchy",    lambda: np.array_equal(a.hierarchy(), b.hierarchy())),
        ("connectivity", lambda: np.array_equal(a.connectivity(), b.connectivity())),
        ("n selective",  lambda: int(a.keep.sum()) == int(b.keep.sum())),
    ]
    # features() is indexed by neurons(), so compare region by region: the two
    # files order neurons the same way, but the check should not depend on that.
    Fa, Fb = a.features("temporal"), b.features("temporal")
    for r in b.regions:
        checks.append((f"features[{r}]",
                       lambda r=r: np.array_equal(Fa[a.neurons(r)], Fb[b.neurons(r)])))

    bad = [name for name, ok in checks if not ok()]
    for name, ok in checks:
        if not ok():
            print(f"  MISMATCH  {name}")
    if bad:
        print(f"\n{len(bad)}/{len(checks)} checks failed -- the slim file is NOT "
              f"equivalent. Re-run extract/posani_selective.py.")
        return 1
    print(f"all {len(checks)} checks pass: {slim.name} "
          f"({slim.stat().st_size / 1e6:.1f} MB) gives the same Dataset as "
          f"{full.name} ({full.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
