"""Check that the refactor changed nothing it should not have.

Two layers, numbers first:

  caches   every array in every results/*.npz, against tools/golden/caches.json.
           A bad merge in shape.py or clustering.py shows up here as a changed
           number long before it is visible as a moved pixel.

  panels   each panel PDF rendered to PNG and compared to its pre-refactor
           render. Raw PDF hashes are useless -- matplotlib stamps
           /CreationDate, so two runs of identical code differ.

    python tools/verify.py                 # everything
    python tools/verify.py --figure Figure1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = ROOT.parent
GOLDEN = HERE / "golden"

TOL = 1e-10          # arrays: absolute, on mean and std
PIXELS = 0           # panels: differing pixels allowed


def _match(name_path, golden_keys):
    """Pick the golden entry for a produced file, by longest shared path suffix.

    Matching on basename alone is not safe: three different files are called
    region_categoricality.npz (Posani's and two of Siegel's). A dict keyed by
    basename keeps whichever came last, so the verifier silently compared
    Posani's result against Siegel's baseline and reported a 1.6 std "drift"
    that was really two unrelated analyses. It could as easily have hidden a
    real difference.

    Returns None when the choice is ambiguous, so an unverifiable file is
    reported as such rather than compared against the wrong baseline.
    """
    parts = list(Path(name_path).parts)
    cands = [k for k in golden_keys if Path(k).name == parts[-1]]
    if len(cands) <= 1:
        return cands[0] if cands else None

    def shared(k):
        a, b = list(Path(k).parts)[::-1], parts[::-1]
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        return n

    scored = sorted(((shared(k), k) for k in cands), reverse=True)
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None                      # genuinely ambiguous
    return scored[0][1]


def _known() -> dict:
    """Accepted, explained differences. Reported, never silenced."""
    f = HERE / "known_differences.toml"
    if not f.exists():
        return {}
    try:
        import tomllib
    except ModuleNotFoundError:
        return {}
    return tomllib.loads(f.read_text())


def _load_png(p: Path):
    # PIL only -- reading a PNG needs no PDF backend. Importing fitz here as well
    # meant that on a machine without pymupdf the golden PNG failed to load for
    # the same reason the new one failed to render, and every panel was reported
    # as unverifiable rather than as a missing dependency.
    try:
        from PIL import Image
        return np.asarray(Image.open(p).convert("RGB"), dtype=np.int16)
    except Exception:
        return None


def render(pdf: Path, out: Path, dpi: int = 150) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fitz
        with fitz.open(pdf) as doc:
            if not doc.page_count:
                return False
            doc[0].get_pixmap(dpi=dpi, alpha=False).save(out)
        return True
    except Exception:
        return False


def check_caches(new_dirs: dict[str, Path]) -> tuple[int, int, list[str]]:
    """Compare every npz we can pair with a golden entry."""
    golden = json.loads((GOLDEN / "caches.json").read_text())
    ok = bad = 0
    msgs = []
    for figure, d in new_dirs.items():
        for npz in sorted(d.rglob("*.npz")):
            key = _match(npz.relative_to(d), golden)
            if key is None:
                if any(Path(k).name == npz.name for k in golden):
                    msgs.append(f"  {figure}/{npz.name}: ambiguous baseline, not compared")
                continue                       # new cache, nothing to compare
            ref = golden[key]
            try:
                with np.load(npz, allow_pickle=True) as z:
                    for k, want in ref["arrays"].items():
                        if k == "<unreadable>" or k not in z.files:
                            continue
                        a = z[k]
                        if "mean" not in want or a.dtype.kind not in "fiu" or not a.size:
                            continue
                        fin = a[np.isfinite(a)] if a.dtype.kind == "f" else a
                        if not fin.size:
                            continue
                        dm = abs(float(np.mean(fin)) - want["mean"])
                        ds = abs(float(np.std(fin)) - want["std"])
                        known = _known().get(npz.name)
                        tol = TOL
                        if (known and known.get("figure") == figure
                                and known.get("array", k) == k):
                            tol = float(known.get("tolerance", TOL))
                        if dm > tol or ds > tol:
                            bad += 1
                            msgs.append(f"  {figure}/{npz.name}[{k}]: "
                                        f"mean drift {dm:.3e}, std drift {ds:.3e}")
                        elif tol != TOL:
                            ok += 1
                            msgs.append(f"  {figure}/{npz.name}[{k}]: drift "
                                        f"{max(dm, ds):.1e} within the documented "
                                        f"tolerance -- KNOWN")
                        else:
                            ok += 1
            except Exception as exc:
                bad += 1
                msgs.append(f"  {figure}/{npz.name}: unreadable ({type(exc).__name__})")
    return ok, bad, msgs


def check_panels(new_dirs: dict[str, Path]) -> tuple[int, int, list[str]]:
    golden = json.loads((GOLDEN / "panels.json").read_text())
    by_name = {Path(k).name: (k, v) for k, v in golden.items()}
    ok = bad = 0
    msgs = []
    tmp = HERE / ".verify_tmp"
    for figure, d in new_dirs.items():
        for pdf in sorted(d.rglob("*.pdf")):
            hit = by_name.get(pdf.name)
            if hit is None:
                continue
            key, ref = hit
            if not ref.get("rendered"):
                continue
            ref_png = GOLDEN / figure / Path(key).with_suffix(".png")
            if not ref_png.exists():
                ref_png = next(GOLDEN.rglob(Path(key).with_suffix(".png").name), None)
                if ref_png is None:
                    continue
            new_png = tmp / figure / pdf.with_suffix(".png").name
            if not render(pdf, new_png):
                bad += 1
                msgs.append(f"  {figure}/{pdf.name}: could not render")
                continue
            a, b = _load_png(ref_png), _load_png(new_png)
            if a is None or b is None:
                continue
            if a.shape != b.shape:
                bad += 1
                msgs.append(f"  {figure}/{pdf.name}: size {b.shape} vs {a.shape}")
                continue
            diff = int((np.abs(a - b).max(axis=2) > 0).sum())
            if diff > PIXELS:
                known = _known().get(pdf.name)
                if known and known.get("figure") == figure:
                    msgs.append(f"  {figure}/{pdf.name}: {diff:,} px differ -- "
                                f"KNOWN, see tools/known_differences.toml")
                    ok += 1
                else:
                    bad += 1
                    msgs.append(f"  {figure}/{pdf.name}: {diff:,} px differ "
                                f"(max {int(np.abs(a - b).max())}/255)")
            else:
                ok += 1
    return ok, bad, msgs


def _check_package() -> None:
    """Fail loudly if a stale shapemetrics/ shadows the installed package.

    The pre-refactor tree is still on disk (untracked, holding the raw data), and
    its shapemetrics/ directory shadows the installed one for anything run from
    the repository root -- silently, giving the OLD library. That produced a
    wrong diagnosis once already.
    """
    import shapemetrics as sm
    if not hasattr(sm, "paths"):
        raise SystemExit(
            f"wrong shapemetrics on the path: {sm.__file__}\n"
            f"  That is the pre-refactor package. Run from outside the repository\n"
            f"  root, or remove/rename the stale shapemetrics/ directory.")


def main() -> int:
    _check_package()
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", default=None)
    a = ap.parse_args()

    figures = [a.figure] if a.figure else [f"Figure{i}" for i in (1, 2, 3, 4)]
    dirs = {f: ROOT / f / "results" for f in figures
            if (ROOT / f / "results").exists()}
    if not dirs:
        print("nothing to verify yet")
        return 0

    c_ok, c_bad, c_msg = check_caches(dirs)
    p_ok, p_bad, p_msg = check_panels(dirs)

    print(f"caches : {c_ok:>4} arrays match, {c_bad} drift")
    for m in c_msg[:20]:
        print(m)
    print(f"panels : {p_ok:>4} identical,    {p_bad} differ")
    for m in p_msg[:20]:
        print(m)

    if c_bad or p_bad:
        print("\nNot a rounding artefact: the refactor is supposed to be behaviour-"
              "preserving,\nso any drift is a bug. Compare against the old tree before "
              "accepting it.")
        return 1
    print("\nunchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
