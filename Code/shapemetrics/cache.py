"""Compute-once-and-cache, so re-running a notebook redraws without recomputing.

The cache records the parameters it was computed under, and refuses to serve
itself to a caller asking for different ones.

That guard is the point. Without it a cache is indistinguishable from a correct
result: a null of 100 draws is served to code that now asks for 500, or a
distance matrix computed at `alpha=1` to code that now passes `alpha=0`, and the
figure renders -- wrong, with no error anywhere. The failure is silent, survives
a re-run, and looks exactly like a real result.
"""
import hashlib
import json

import numpy as np

FINGERPRINT = "_params"          # reserved key inside every cached npz


def _fingerprint(params) -> str:
    """A stable digest of the parameters a cache was computed under."""
    if params is None:
        return ""
    blob = json.dumps(params, sort_keys=True, default=repr)
    return hashlib.sha256(blob.encode()).hexdigest()[:16] + "|" + blob


def cached_npz(path, compute, valid=None, params=None):
    """Return the arrays at `path`, computing and saving them first if needed.

    `compute` returns a dict of arrays.

    `params` is a dict of whatever would change the answer -- draw counts, alpha,
    n_pcs, seeds. It is stored alongside the arrays and compared on load; a
    mismatch recomputes rather than returning a stale result. Passing nothing
    keeps the old permissive behaviour, so existing callers still work, but new
    code should pass it.

    `valid` is an optional predicate on the loaded dict for anything a parameter
    digest cannot express.
    """
    want = _fingerprint(params)
    if path.exists():
        d = dict(np.load(path, allow_pickle=True))
        got = d.get(FINGERPRINT)
        got = got.item() if isinstance(got, np.ndarray) else got
        got = got.decode() if isinstance(got, bytes) else got
        stale = params is not None and got != want
        if not stale and (valid is None or valid(d)):
            d.pop(FINGERPRINT, None)
            return d
        if stale:
            print(f"cache {path.name}: parameters changed, recomputing\n"
                  f"    cached: {(got or '<none recorded>')[17:] or '<none>'}\n"
                  f"    wanted: {want[17:]}")
    out = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out, **({FINGERPRINT: want} if params is not None else {}))
    return out
