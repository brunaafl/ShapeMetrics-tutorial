"""Where the data is. The only module in the package that knows.

Before this, finding data meant a `sys.path.insert` and a literal path in every
notebook -- 18 of the former and 7 of the latter, one naming `/Users/jbarbosa/`,
a username that does not exist on this machine, so it was already broken.

Data comes in two tiers, and the distinction is the point:

    derived   Data/derived/<paper>/     tracked in git, always present (177 MB)
    raw       wherever it actually is   67 GB, never copied

**A notebook reads only the derived tier plus its own results/.** Anything
reaching for raw data is an extraction script wearing a notebook costume and
belongs in `extract/`. That rule is what lets a fresh clone rebuild every figure
without the 67 GB, and `tools/check_standalone.py` is what keeps it true.

There used to be a third tier between them -- derived files too big for git,
resolved through data_roots.toml -- and four notebooks read it, so those four
could not run from a clone at all. Those files are now either slimmed
(rrr_neurons.npz, 178 MB -> 14 MB, losing nothing any analysis reads) or simply
tracked. `external()` remains, for `extract/` and for the four Figure 3 panels
that analyse the raw .mat sessions directly.

Roots resolve in this order, so a collaborator overrides without editing code:

    1. $SHAPEMETRICS_DATA
    2. data_roots.toml       (gitignored; data_roots.example.toml is the template)
    3. Data/external         (the default)

Typical use:

    from shapemetrics import paths
    paths.set_figure("Figure3")
    D   = paths.derived("duszkiewicz2024", "hd_tuning.npz")
    out = paths.results("hd_clustering.npz")
    f   = paths.cache("siegel2015", "pooled_neurons.npz")   # results/, else tracked
"""
from __future__ import annotations

import os
from pathlib import Path

try:                                     # 3.11+; the envs pin 3.10 for allensdk
    import tomllib
except ModuleNotFoundError:              # pragma: no cover
    tomllib = None

ROOT = Path(__file__).resolve().parents[2]      # refactored-code/
DATA = ROOT / "Data"
DERIVED = DATA / "derived"
EXTERNAL = DATA / "external"

# Filled by set_figure(); results() refuses to guess.
_FIGURE: str | None = None

#: dataset id -> (source paper, what it is). The ids name Data/ subdirectories
#: and mirror the citation keys in the manuscript's refs.bib.
DATASETS = {
    "posani2026":         ("Posani et al. 2026", "RRR encoding-model coefficients, IBL BWM"),
    "angelaki2025brain":  ("IBL / Angelaki et al. 2025", "Brain-Wide Map spikes"),
    "harris2019":         ("Harris et al. 2019", "Allen mouse cortico-cortical connectivity"),
    "siegel2015":         ("Siegel, Buschman & Miller 2015", "monkey multi-area recordings"),
    "duszkiewicz2024":    ("Duszkiewicz et al. 2024", "mouse postsubiculum head-direction cells"),
    "noel2025":           ("Noel et al. 2025", "autism mouse lines, GAM kernels"),
    "iblreproducibility": ("IBL 2024", "repeated-site Neuropixels + behaviour"),
    "Siegle2021-hs":      ("Siegle et al. 2021", "Allen Brain Observatory (Methods only)"),
}


class MissingDataset(FileNotFoundError):
    """Raised with enough context to act on, rather than a bare path."""


def _resolve(value: str) -> Path:
    """Expand ~ and anchor a relative root at refactored-code/, not the cwd.

    Relative entries are natural to write ("../Posani") but would otherwise
    resolve against wherever the notebook was launched -- so the same config
    would find the data from one directory and not from another.
    """
    p = Path(value).expanduser()
    return p if p.is_absolute() else (ROOT / p).resolve()


def _roots() -> dict[str, Path]:
    """Per-dataset roots from data_roots.toml, plus its `default`."""
    cfg = ROOT / "data_roots.toml"
    if not cfg.exists() or tomllib is None:
        return {}
    conf = tomllib.loads(cfg.read_text())
    out = {k: _resolve(v) for k, v in conf.get("roots", {}).items()}
    if "default" in conf:
        out["__default__"] = _resolve(conf["default"])
    return out


def external_root(dataset: str) -> Path:
    """The directory holding `dataset`'s raw and large-derived files."""
    if env := os.environ.get("SHAPEMETRICS_DATA"):
        return Path(env).expanduser().resolve() / dataset
    roots = _roots()
    if dataset in roots:
        return roots[dataset]
    if "__default__" in roots:
        return roots["__default__"] / dataset
    return EXTERNAL / dataset


def _describe(dataset: str) -> str:
    paper, what = DATASETS.get(dataset, ("unknown", "unknown"))
    return f"{dataset} ({paper}: {what})"


def derived(dataset: str, *parts: str) -> Path:
    """A small, git-tracked derived file. Present in a fresh clone."""
    p = DERIVED.joinpath(dataset, *parts)
    if not p.exists():
        raise MissingDataset(
            f"{_describe(dataset)}\n"
            f"  expected: {p}\n"
            f"  This tier is tracked in git, so its absence means either a bad "
            f"path or an incomplete checkout. See Data/README.md.")
    return p


def external(dataset: str, *parts: str) -> Path:
    """Raw or large-derived data. May legitimately be absent on this machine."""
    p = external_root(dataset).joinpath(*parts)
    if not p.exists():
        raise MissingDataset(
            f"{_describe(dataset)}\n"
            f"  expected: {p}\n"
            f"  Not tracked in git -- it is raw or large-derived data. Either point\n"
            f"  {ROOT / 'data_roots.toml'} at it, set $SHAPEMETRICS_DATA, or\n"
            f"  regenerate it with the script named for this dataset in "
            f"{ROOT / 'extract'}.\n"
            f"  Sizes and sources: Data/README.md")
    return p


def cache(dataset: str, *parts: str) -> Path:
    """A recomputable intermediate: this figure's results/ first, the tracked tier second.

    Panel notebooks are written as

        f = OUT / "region_categoricality.npz"
        if f.exists():  load(f)
        else:           compute(); np.savez(f, ...)

    which on a fresh clone always takes the `else` branch -- and the `else`
    branch is the one that needs the data that is not in the clone. The
    computed answers, though, ARE tracked, under Data/derived/<dataset>/. So
    look there before deciding nothing is cached:

        f = paths.cache("siegel2015", "region_categoricality.npz")

    Returns the results/ path when that file exists (a local recomputation wins,
    so deleting it still forces a rebuild), the tracked path when only that
    exists, and otherwise the results/ path -- which does not exist yet, so the
    caller computes and writes there, never into Data/derived.
    """
    p = results(*parts)
    if p.exists():
        return p
    try:
        return derived(dataset, *parts)
    except MissingDataset:
        return p


def set_figure(name: str) -> Path:
    """Point results() at one figure's folder. Call once, at the top of a notebook."""
    global _FIGURE
    if not (ROOT / name).is_dir():
        raise ValueError(f"no such figure folder: {ROOT / name}")
    _FIGURE = name
    (ROOT / name / "results").mkdir(parents=True, exist_ok=True)
    return ROOT / name / "results"


def results(*parts: str) -> Path:
    """A path inside the current figure's results/. Caches and panels go here."""
    if _FIGURE is None:
        raise RuntimeError(
            "call paths.set_figure('Figure1') first -- results are figure-scoped, "
            "which is what stops two notebooks writing the same file (region_space "
            "and theta_space both wrote results/region_examples.* before this).")
    p = (ROOT / _FIGURE / "results").joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def figure_code(figure: str | None = None):
    """Import a figure's local `code/` package, safely.

    `code` is a standard-library module name. A notebook doing
    `from code import region_space` gets our package only when the working
    directory happens to be the figure folder, and the *stdlib* `code` module
    otherwise -- silently, with a confusing AttributeError rather than an
    ImportError. Since the folder name is part of the agreed layout, load it by
    path under a unique name instead:

        sim = paths.figure_code()          # after set_figure(...)
        X, region, colour = sim.region_space.scenario("kinds")
    """
    import importlib.util
    import sys as _sys

    figure = figure or _FIGURE
    if figure is None:
        raise RuntimeError("pass a figure name, or call set_figure() first")
    init = ROOT / figure / "code" / "__init__.py"
    if not init.exists():
        raise FileNotFoundError(f"{figure} has no code/ package: {init}")

    name = f"_{figure.lower()}_code"
    if name in _sys.modules:
        return _sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, init, submodule_search_locations=[str(init.parent)])
    module = importlib.util.module_from_spec(spec)
    _sys.modules[name] = module            # before exec, so relative imports work
    spec.loader.exec_module(module)
    return module


def require_env(name: str) -> None:
    """Fail fast, and usefully, when a script is run in the wrong conda env.

    The three environments are deliberately separate -- allensdk pins
    numpy<1.24 and ibllib pulls ~70 packages -- so this is a routine mistake
    with an unhelpful error (an ImportError deep in a dependency).
    """
    current = os.environ.get("CONDA_DEFAULT_ENV")
    if current != name:
        raise RuntimeError(
            f"this script needs the '{name}' environment, but "
            f"CONDA_DEFAULT_ENV={current!r}.\n"
            f"    conda activate {name}\n"
            f"or: conda run -n {name} python {os.path.basename(__import__('sys').argv[0])}")
