"""Copy a notebook into the new layout and rewrite how it finds code and data.

Only the plumbing changes. The analysis cells are copied verbatim, so a ported
notebook must reproduce its cached numbers exactly -- which is what
tools/verify.py then checks.

What it rewrites:

  * the `HERE = next(p for p in ...)` / `sys.path.insert` header, which existed
    only because nothing was installed, into `from shapemetrics import paths`
  * `OUT = HERE / "results"` into `OUT = paths.set_figure(...)`, so results are
    figure-scoped -- two notebooks used to write results/region_examples.* and
    whichever ran last won
  * `import region_space as R` into a load through paths.figure_code(), because
    `code` is a stdlib module name and a bare import silently resolves to the
    standard library from any directory but the figure's own

    python tools/port_notebook.py <src.ipynb> <Figure1> [--name figure1.ipynb]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# module name -> attribute on the figure's code package
SIM_MODULES = ("region_space", "theta_space", "grid_space", "simulation",
               "panels", "hd_panels", "bwm_panels", "siegel_panels", "ibl_panels")


def rewrite_header(src: str, figure: str) -> tuple[str, bool]:
    """Replace the path-discovery preamble. Returns (text, changed)."""
    original = src
    lines, out, skip = src.splitlines(keepends=True), [], False
    for ln in lines:
        st = ln.strip()
        # the multi-line `HERE = next(p for p in (...) if (...).exists())` block:
        # nested parens defeat a regex, so consume it line by line
        if st.startswith("HERE = next("):
            skip = True
        if skip:
            if st.endswith(".exists())") or st.endswith("Path.cwd()"):
                skip = False
            continue
        if st.startswith("sys.path.insert(") or st.startswith("REPO = Path("):
            continue
        if st.startswith("HERE = "):          # any other HERE assignment
            continue
        out.append(ln)
    src = "".join(out)

    # results directory -> figure-scoped
    src = re.sub(r'OUT = (?:HERE|Path\.cwd\(\)) / "results"\n'
                 r'(?:OUT\.mkdir\([^\n]*\)\n)?',
                 f'OUT = paths.set_figure("{figure}")\n', src)
    if "paths.set_figure" not in src:
        src = re.sub(r"^(from shapemetrics import paths)$",
                     f'\\1\n\nOUT = paths.set_figure("{figure}")',
                     src, count=1, flags=re.M)

    # sibling simulation modules -> the figure's code package.
    # trailing "# noqa" comments are common, so match the statement not the line.
    # Self-contained, so the rewritten line does not depend on where it lands.
    for m in SIM_MODULES:
        pat = rf"^import {m}(?: as (\w+))?(\s*#.*)?$"
        mt = re.search(pat, src, flags=re.M)
        if mt:
            alias = mt.group(1) or m
            src = re.sub(pat, f'{alias} = paths.figure_code("{figure}").{m}',
                         src, flags=re.M)

    if "from shapemetrics import paths" not in src:
        src = re.sub(r"^(import numpy as np)$",
                     "from shapemetrics import paths\n\\1", src, count=1, flags=re.M)

    # `import sys` is usually dead once the path juggling is gone
    if not re.search(r"\bsys\.", src.replace("import sys", "")):
        src = re.sub(r"^import sys\n", "", src, flags=re.M)

    return src, src != original


def port(src_path: Path, figure: str, name: str | None,
         dest_dir: str | None = None) -> Path:
    nb = json.loads(src_path.read_text())
    changed = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"])
        new, did = rewrite_header(text, figure)
        if did:
            cell["source"] = new.splitlines(keepends=True)
            changed += 1
        cell["outputs"] = []              # ported notebooks are re-run, not trusted
        cell["execution_count"] = None

    dest = ROOT / (dest_dir or figure) / (name or src_path.name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(nb, indent=1))
    print(f"  {src_path.name:<28} -> {dest.relative_to(ROOT)}  ({changed} cell(s) rewritten)")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("figure")
    ap.add_argument("--name", default=None)
    ap.add_argument("--dest", default=None,
                    help="destination dir, if not the figure folder itself "
                         "(panels/ still writes into the figure's results/)")
    a = ap.parse_args()
    if not a.src.exists():
        print(f"no such notebook: {a.src}", file=sys.stderr)
        return 1
    port(a.src, a.figure, a.name, a.dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
