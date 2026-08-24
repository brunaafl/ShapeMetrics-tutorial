"""Per-subject psychometric curves for the Repeated-Site sessions.

The companion to `extract_bwm_tuning.py`: that script keeps each neuron's 90-dim
tuning curve, this one keeps each session's behaviour, so the two can be related
subject by subject.

Trial selection matches the neural script exactly -- finite alignment and
contrast, 50-50 blocks excluded -- so the trials behind a subject's psychometric
curve are the trials behind its tuning curves.

The curve is P(choice = rightward | signed contrast); in IBL's coding the
rightward choice is choice = -1, so the curve rises with signed contrast over the same 9 contrasts,
a 9-vector per session.  Trials with no choice (choice = 0) are dropped.

    python extract_bwm_behavior.py          # writes data_bwm/bwm_behavior.npz
"""
from pathlib import Path
import sys
import traceback

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "ibl_analyses/scripts"))

from one.api import ONE                                       # noqa: E402
from setup import good_eids, params as SETUP                  # noqa: E402

CACHE = REPO / "ibl_analyses/data" / SETUP["tag"]
OUT = Path(__file__).resolve().parent / "data_bwm"
CONTRASTS = [-100.0, -25.0, -12.5, -6.25, 0.0, 6.25, 12.5, 25.0, 100.0]


def signed_contrast(trials):
    return np.diff(np.nan_to_num(
        np.c_[trials["contrastLeft"], trials["contrastRight"]])).ravel() * 100


def session_curve(one, eid):
    trials = one.load_object(eid, "trials", collection="alf")
    contrast = signed_contrast(trials)
    align = trials["response_times"] if SETUP["align_to"] == "response" \
        else trials["stimOn_times"]
    choice = np.asarray(trials["choice"])
    valid = (np.isfinite(align) & np.isfinite(contrast)
             & (trials["probabilityLeft"] != 0.5)      # same mask as the neural script
             & (choice != 0))                          # no-choice trials dropped
    contrast, choice = contrast[valid], choice[valid]

    p, n = [], []
    for c in CONTRASTS:
        m = np.isclose(contrast, c)
        n.append(int(m.sum()))
        p.append(float(np.mean(choice[m] == -1)) if m.any() else np.nan)
    if np.any(~np.isfinite(p)):
        return None                                    # a contrast is missing
    return np.array(p), np.array(n)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    one = ONE(cache_dir=CACHE, mode="local")
    rows = []
    for i, eid in enumerate(good_eids, 1):
        try:
            got = session_curve(one, eid)
        except Exception:
            print(f"[{i}/{len(good_eids)}] {eid[:8]} FAILED", flush=True)
            traceback.print_exc(limit=1)
            continue
        if got is None:
            print(f"[{i}/{len(good_eids)}] {eid[:8]} unusable", flush=True)
            continue
        p, n = got
        try:
            subject = one.get_details(eid)["subject"]
        except Exception:
            subject = eid[:8]
        rows.append(dict(p=p, n=n, subject=subject, eid=eid))
        print(f"[{i}/{len(good_eids)}] {subject:<12} {n.sum():>5} trials   "
              f"p(right) {p[0]:.2f} -> {p[-1]:.2f}", flush=True)

    if not rows:
        sys.exit("no sessions read -- check the cache path")
    out = OUT / "bwm_behavior.npz"
    np.savez_compressed(
        out,
        psychometric=np.stack([r["p"] for r in rows]),
        n_trials=np.stack([r["n"] for r in rows]),
        subject=np.array([r["subject"] for r in rows]),
        eid=np.array([r["eid"] for r in rows]),
        contrasts=np.array(CONTRASTS))
    print(f"\nwrote {out}:  {len(rows)} sessions, "
          f"{len(set(r['subject'] for r in rows))} subjects")


if __name__ == "__main__":
    main()
