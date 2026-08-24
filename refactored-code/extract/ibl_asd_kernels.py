"""Collect the per-neuron GAM kernels of the IBL autism dataset into one array.

The release stores one .mat per neuron under <genotype>/<region>/, each holding a
`results` struct with one entry per model covariate. Two things make a naive read
unusable, and both are handled here:

  * the covariate set varies between neurons -- 17 are almost universal, but
    `choice0`, `prior80`, `movement_PC1..10` and others appear in a minority;
  * the same covariate can have a 100- or a 106-sample kernel depending on the
    neuron, so concatenating by position would silently misalign features.

So covariates are matched by NAME, and a neuron is kept only if it has all of
CORE_VARS at the expected length. That gives every kept neuron the same
954-dimensional feature vector: 9 stimulus kernels x 106 time bins.

    python extract_asd_kernels.py            # writes data_asd/asd_kernels.npz
"""

from collections import defaultdict
from pathlib import Path
import glob
import sys

import numpy as np
import scipy.io as sio

SRC = Path("/Users/joaobarbosa/Dropbox/IBL_ASD_data")
OUT = Path(__file__).resolve().parent / "data_asd"

# The nine stimulus kernels: one per signed contrast, each 106 time bins. Every
# neuron in the release has all nine at this length. The other covariates are left
# out -- `spike_hist` is an autoregressive nuisance term, and the length of
# `subjective_prior`, `prior20`, `prev_choiceL` and `prev_feedback_correct` varies
# between neurons, so they cannot be concatenated into a fixed feature vector.
CORE_VARS = ["cL100", "cL025", "cL012", "cL006", "c0000",
             "cR006", "cR012", "cR025", "cR100"]
KERNEL_LEN = 106
GENOTYPES = {"N": "wild type", "F": "Fmr1", "C": "Cntnap2", "S": "Shank3"}


def load_results(path):
    """The `results` struct of one neuron file, whatever format it was saved in.

    A handful of files are numpy structured arrays that were given a .mat suffix,
    so both readers are tried before giving up.
    """
    try:
        return np.atleast_1d(
            sio.loadmat(path, squeeze_me=True, struct_as_record=False)["results"])
    except Exception:
        pass
    try:
        a = np.load(path, allow_pickle=True)
        return np.atleast_1d(a)
    except Exception:
        return None


def field(entry, name):
    """Field access that works for both mat_struct and numpy void records."""
    try:
        return getattr(entry, name)
    except AttributeError:
        return entry[name]


def neuron_record(path, genotype):
    res = load_results(path)
    if res is None:
        return None

    kernels, strengths = {}, {}
    for e in res:
        try:
            v = str(field(e, "variable"))
            k = np.atleast_1d(np.asarray(field(e, "kernel"), dtype=float)).ravel()
        except Exception:
            return None
        kernels[v] = k
        try:
            strengths[v] = float(np.ravel(field(e, "kernel_strength"))[0])
        except Exception:
            strengths[v] = np.nan

    if any(v not in kernels or kernels[v].shape[0] != KERNEL_LEN for v in CORE_VARS):
        return None
    if not np.all(np.isfinite(np.concatenate([kernels[v] for v in CORE_VARS]))):
        return None

    e0 = res[0]
    def meta(name, default=""):
        try:
            return str(field(e0, name))
        except Exception:
            return default

    return dict(
        x=np.concatenate([kernels[v] for v in CORE_VARS]).astype(np.float32),
        strength=np.array([strengths[v] for v in CORE_VARS], np.float32),
        genotype=genotype,
        region=meta("brain_area_group"),
        animal=meta("animal_name"),
        date=meta("date"),
        neuron_id=meta("neuron_id"),
        fr=float(np.ravel(field(e0, "fr"))[0]) if hasattr(e0, "fr") else np.nan,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], defaultdict(int)

    for g in GENOTYPES:
        files = sorted(glob.glob(str(SRC / g / "*" / "gam_fit_useCoupling0*")))
        kept = 0
        for i, f in enumerate(files, 1):
            rec = neuron_record(f, g)
            if rec is None:
                skipped[g] += 1
                continue
            rows.append(rec)
            kept += 1
            if i % 1000 == 0:
                print(f"  {g}: {i}/{len(files)}", flush=True)
        print(f"{g} ({GENOTYPES[g]:<10}): {kept} kept, {skipped[g]} skipped "
              f"of {len(files)}", flush=True)

    if not rows:
        sys.exit("no neurons read -- check SRC")

    X = np.stack([r["x"] for r in rows])
    out = OUT / "asd_kernels.npz"
    np.savez_compressed(
        out,
        X=X,
        strength=np.stack([r["strength"] for r in rows]),
        genotype=np.array([r["genotype"] for r in rows]),
        region=np.array([r["region"] for r in rows]),
        animal=np.array([r["animal"] for r in rows]),
        date=np.array([r["date"] for r in rows]),
        neuron_id=np.array([r["neuron_id"] for r in rows]),
        fr=np.array([r["fr"] for r in rows]),
        variables=np.array(CORE_VARS),
        kernel_len=KERNEL_LEN,
    )
    animals = np.array([r["animal"] for r in rows])
    print(f"\nwrote {out}")
    print(f"  {X.shape[0]} neurons x {X.shape[1]} features "
          f"({len(CORE_VARS)} covariates x {KERNEL_LEN} samples)")
    print(f"  {len(np.unique(animals))} animals, "
          f"{len(np.unique([r['region'] for r in rows]))} regions")


if __name__ == "__main__":
    main()
