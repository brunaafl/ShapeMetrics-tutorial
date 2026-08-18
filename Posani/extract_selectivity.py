"""
Step 1: stream the (1.3 GB) RRR_selectivity.json released by Posani, Wang et al. (2026)
and save a compact .npz with, for every neuron:
    acronym   brain region
    eid       session id
    dR2       RRR_r2 - null_r2 (selectivity criterion used in the paper)
    beta      time-varying encoding coefficients, shape (n_vars, n_timesteps)

Selectivity (Fig. 2 of the paper) is alpha = sum_t |beta[v, t]|.
"""
import numpy as np
import ijson

SRC = "brainwide-RRR-encoding-model/trained_RRR_model/RRR_selectivity.json"
OUT = "rrr_neurons.npz"

VAR_LIST = ["block", "side", "contrast", "choice", "outcome", "wheel", "whisker", "lick"]


def column(name, transform=lambda v: v):
    """Stream one top-level column of the column-oriented json."""
    out = {}
    with open(SRC, "rb") as f:
        for rowid, value in ijson.kvitems(f, name, use_float=True):
            out[rowid] = transform(value)
    print(f"  {name}: {len(out)} rows")
    return out


print("streaming acronym / RRR_r2 / null_r2 ...")
acronym = column("acronym")
r2 = column("RRR_r2")
null_r2 = column("null_r2")

print("streaming RRR_beta (drop the bias row, keep n_vars x n_timesteps) ...")
beta = column("RRR_beta", lambda v: np.asarray(v, dtype=np.float32)[:-1])

rows = sorted(acronym, key=int)
np.savez_compressed(
    OUT,
    acronym=np.array([acronym[r] for r in rows]),
    dR2=np.array([r2[r] - null_r2[r] for r in rows], dtype=np.float32),
    beta=np.stack([beta[r] for r in rows]),
    var_list=np.array(VAR_LIST),
)
print(f"saved {OUT}: {len(rows)} neurons, beta shape {beta[rows[0]].shape}")
