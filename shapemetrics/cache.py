"""Compute-once-and-cache, so re-running a notebook redraws without recomputing."""
import numpy as np


def cached_npz(path, compute, valid=None):
    """Return the arrays at `path`, computing and saving them first if needed.

    `compute` returns a dict of arrays. `valid` is an optional predicate on the
    loaded file: return False and the result is recomputed rather than reused --
    the guard that keeps a null of 100 draws from being served when the notebook
    now asks for 500.
    """
    if path.exists():
        d = dict(np.load(path, allow_pickle=True))
        if valid is None or valid(d):
            return d
    out = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return out
