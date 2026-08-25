"""Check that a fresh clone can run everything, without downloading anything.

The repository's promise is one sentence: *no notebook reads outside
`Data/derived/` and its own `results/`.* That promise is easy to break by
accident -- one `paths.external(...)` added to a notebook, or one tracked file
quietly not committed -- and the breakage is invisible on a machine that happens
to have the 67 GB raw tree sitting next to it, which is the machine the notebooks
are written on. That is exactly how the four notebooks this script was written
for came to be broken.

Two checks, both static, so neither needs the raw data or a kernel:

  reachable   every `paths.derived(dataset, "name")` in a notebook or in figure
              code names a file that exists. A missing one is a file that was
              produced but never committed.

  external    no notebook calls `paths.external(...)`, opens an absolute path,
              or opens a bare relative one -- except the notebooks in EXCEPTIONS,
              which are exploratory analyses over raw .mat that no derived file
              covers.

  cwd         nothing writes to a location that depends on where the kernel was
              started: no `Path.cwd()`, no literal `folder=`. This one is about
              output rather than input, and it is here because the same run that
              proved the figures reproduce also put three panels in
              Figure2/panels/results/ -- a directory nothing reads.

    python tools/check_standalone.py

Exit status is 1 if anything fails, so it can gate a commit.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DERIVED = ROOT / "Data" / "derived"

#: Notebooks allowed to read the raw tier: they analyse the .mat sessions
#: directly, including Dataset_3 and the CellTypes files, and no derived
#: product stands in for that. None of them feeds a compiled figure.
EXCEPTIONS = {
    "Figure3/panels/canonical_circuit.ipynb",
    "Figure3/panels/inhibitory_analyses.ipynb",
    "Figure3/panels/inter_animal_variability.ipynb",
    "Figure3/panels/rotation_Cue.ipynb",
}

#: Notebooks whose raw-tier read is a FALLBACK: they load a tracked cache and
#: only rebuild it from raw when it is absent. The clone path never gets there.
FALLBACK = {
    "Figure2/code/data.py",       # _coefficients(): tracked slim file, else the 178 MB one
    "Figure3/code/hd.py",         # tuning_curves(rebuild=True)
    "Figure3/panels/animal_decoding.ipynb",   # hd_curves.npz, else the raw sessions
}

READERS = {"open", "loadmat", "read_csv", "load", "save", "savez", "savez_compressed",
           "savefig", "imread", "imsave", "to_csv"}


def _call_name(node: ast.Call) -> str:
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return ""


def _is_paths(node: ast.Call, attr: str) -> bool:
    f = node.func
    return (isinstance(f, ast.Attribute) and f.attr == attr
            and isinstance(f.value, ast.Name) and f.value.id == "paths")


def _bad_literal(s: str) -> bool:
    """An absolute path, or a relative one that depends on the working directory."""
    return s.startswith("/") or s.startswith("../") or bool(re.match(r"^\w+/", s))


def sources(path: Path):
    """Every code string in a file: notebook cells, or the whole .py."""
    if path.suffix == ".ipynb":
        nb = json.loads(path.read_text())
        return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    return [path.read_text()]


def scan(src: str):
    """(derived_refs, uses_external, bad_paths) from real code only.

    Parsed, not grepped: a path named in a comment or quoted inside a docstring
    is not a read, and treating it as one made this script report six commented-out
    savefigs and one line of its own documentation.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return [], False, []                # IPython magics etc.: skip the cell
    refs, external, bad = [], False, []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Path.cwd(): an output location that depends on where Jupyter started.
        # Three golden-referenced Siegel panels were written to
        # Figure2/panels/results/ this way, where verify.py never looked.
        if _call_name(node) == "cwd":
            bad.append((getattr(node, "lineno", 0), "Path.cwd()"))
        # folder="results" overrode plotting.save()'s paths.results() default
        # with a relative path -- the very thing that default exists to prevent.
        for kw in node.keywords:
            if kw.arg in ("folder", "out_dir", "outdir") and \
                    isinstance(kw.value, ast.Constant) and \
                    isinstance(kw.value.value, str):
                bad.append((getattr(node, "lineno", 0), f"{kw.arg}={kw.value.value!r}"))
        if _is_paths(node, "derived") or _is_paths(node, "cache"):
            args = [a.value for a in node.args if isinstance(a, ast.Constant)
                    and isinstance(a.value, str)]
            if len(args) >= 2:
                refs.append((args[0], "/".join(args[1:])))
        if _is_paths(node, "external"):
            external = True
        if _call_name(node) in READERS:
            for a in node.args[:1]:
                if isinstance(a, ast.Constant) and isinstance(a.value, str) \
                        and _bad_literal(a.value):
                    bad.append((getattr(node, "lineno", 0), a.value))
    return refs, external, bad


def main() -> int:
    targets = sorted(ROOT.glob("Figure*/**/*.ipynb")) + sorted(ROOT.glob("Figure*/code/*.py"))
    targets = [t for t in targets if ".ipynb_checkpoints" not in t.parts]

    missing, external, bad_paths = [], [], []
    n_refs = 0
    for t in targets:
        rel = t.relative_to(ROOT).as_posix()
        for src in sources(t):
            refs, uses_external, bad = scan(src)
            for ds, name in refs:
                n_refs += 1
                if not (DERIVED / ds / name).exists():
                    missing.append(f"{rel}: Data/derived/{ds}/{name}")
            if rel in EXCEPTIONS or rel in FALLBACK:
                continue
            if uses_external:
                external.append(rel)
            for lineno, literal in bad:
                bad_paths.append(f"{rel}: {literal!r}")

    print(f"reachable : {n_refs} tracked-tier references, {len(missing)} missing")
    for m in missing:
        print(f"  MISSING  {m}")
    print(f"external  : {len(set(external))} notebook(s) reach for the raw tier "
          f"outside the {len(EXCEPTIONS)} documented exceptions")
    for e in sorted(set(external)):
        print(f"  RAW      {e}")
    print(f"hardcoded : {len(bad_paths)} absolute or bare-relative path(s)")
    for b in bad_paths:
        print(f"  PATH     {b}")

    if missing or external or bad_paths:
        print("\nA clone would not reproduce this. Route the read through "
              "paths.derived / paths.cache, or commit the file it names.")
        return 1
    print("\nstandalone: every notebook outside the documented exceptions reads "
          "only the tracked tier")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
