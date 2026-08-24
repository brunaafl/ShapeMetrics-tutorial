"""Record what the analyses produce TODAY, before anything moves.

Nothing in the refactor is verifiable without this. Two snapshots:

  panels   every panel PDF rendered to PNG at 150 dpi.  Raw PDF hashes are
           useless -- matplotlib stamps /CreationDate, so two runs of identical
           code differ -- and a rendered image is what a reader actually sees.

  caches   sha256 and array-level statistics of every results/*.npz, which is a
           far finer net than any image diff: a wrong merge in shape.py or
           clustering.py shows up here as a changed number long before it is
           visible as a moved pixel.

    python tools/make_golden.py            # writes tools/golden/
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
GOLDEN = HERE / "golden"

# The directories that feed the four compiled figures.  Anything else is out of
# scope (see the plan): allen-brain, fmri, metric-comparison, simple-ring.
SOURCES = {
    "Figure1": ["clustering-simulation/results", "schematic"],
    "Figure2": ["Posani/results", "Posani/figures", "siegel_analyses/results"],
    "Figure3": ["duszkiewicz_analyses/results"],
    "Figure4": ["new_IBL_analyses/figure_ibl", "new_IBL_analyses/results_asd",
                "new_IBL_analyses/results_bwm", "new_IBL_analyses/results_regression",
                "new_IBL_analyses/data_fig"],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def render(pdf: Path, out_stem: Path, dpi: int = 150) -> bool:
    """PDF -> PNG. PyMuPDF if present (it is, in the shapemetrics env), else pdftoppm."""
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    png = out_stem.with_suffix(".png")
    try:
        import fitz                                    # PyMuPDF
        with fitz.open(pdf) as doc:
            if not doc.page_count:
                return False
            doc[0].get_pixmap(dpi=dpi, alpha=False).save(png)
        return png.exists()
    except ImportError:
        pass
    except Exception:
        return False
    try:
        subprocess.run(
            ["pdftoppm", "-r", str(dpi), "-png", "-singlefile",
             str(pdf), str(out_stem)],
            check=True, capture_output=True, timeout=120)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError):
        return False
    return png.exists()


def summarise_npz(path: Path) -> dict:
    """sha256 plus per-array shape/dtype/finite-mean, so a diff says WHICH array moved."""
    entry = {"sha256": sha256(path), "bytes": path.stat().st_size, "arrays": {}}
    try:
        with np.load(path, allow_pickle=True) as z:
            for k in z.files:
                a = z[k]
                rec = {"shape": list(np.shape(a)), "dtype": str(a.dtype)}
                if a.dtype.kind in "fiu" and a.size:
                    finite = a[np.isfinite(a)] if a.dtype.kind == "f" else a
                    if finite.size:
                        rec["mean"] = float(np.mean(finite))
                        rec["std"] = float(np.std(finite))
                entry["arrays"][k] = rec
    except Exception as exc:                       # object arrays, pickles, etc.
        entry["arrays"] = {"<unreadable>": f"{type(exc).__name__}: {exc}"}
    return entry


def main() -> int:
    try:
        import fitz                                    # noqa: F401
        print("renderer: PyMuPDF")
    except ImportError:
        if subprocess.run(["which", "pdftoppm"], capture_output=True).returncode:
            print("WARNING: no renderer (PyMuPDF or pdftoppm) -- panels hashed only.")

    panels, caches, skipped = {}, {}, []
    for figure, dirs in SOURCES.items():
        for rel in dirs:
            root = REPO / rel
            if not root.exists():
                skipped.append(rel)
                continue
            for pdf in sorted(root.rglob("*.pdf")):
                key = str(pdf.relative_to(REPO))
                stem = GOLDEN / figure / pdf.relative_to(REPO).with_suffix("")
                rec = {"sha256": sha256(pdf), "bytes": pdf.stat().st_size}
                rec["rendered"] = render(pdf, stem)
                panels[key] = rec
            for npz in sorted(root.rglob("*.npz")):
                caches[str(npz.relative_to(REPO))] = summarise_npz(npz)

    GOLDEN.mkdir(parents=True, exist_ok=True)
    (GOLDEN / "panels.json").write_text(json.dumps(panels, indent=1, sort_keys=True))
    (GOLDEN / "caches.json").write_text(json.dumps(caches, indent=1, sort_keys=True))

    n_rendered = sum(1 for v in panels.values() if v["rendered"])
    print(f"panels : {len(panels)} pdf, {n_rendered} rendered to png")
    print(f"caches : {len(caches)} npz, "
          f"{sum(len(v['arrays']) for v in caches.values())} arrays")
    if skipped:
        print(f"skipped (absent): {', '.join(skipped)}")
    print(f"-> {GOLDEN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
