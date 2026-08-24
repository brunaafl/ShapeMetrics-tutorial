"""Loading and grouping the Posani, Wang et al. (2026) RRR encoding model."""
import numpy as np
import pandas as pd
from pathlib import Path

from shapemetrics import paths

# rrr_neurons.npz is 178 MB -- the large-derived tier, found via data_roots.toml.
# The two CSVs are small and tracked, so they come from Data/derived/.
# See Data/README.md; they were previously read out of a git clone nested inside
# the analysis directory.

VAR_LIST = ["block", "side", "contrast", "choice", "outcome", "wheel", "whisker", "lick"]
DR2_THRESHOLD = 0.015    # selective-neuron criterion used in the paper
MIN_NEURONS = 50         # minimum selective neurons per region (their Fig. 2d)

MODULES = {              # coarse cortical groupings, for colouring only
    "somatosensory": ("SSp", "SSs"), "visual": ("VIS",), "auditory": ("AUD",),
    "retrosplenial": ("RSP",), "motor": ("MOp", "MOs"),
    "prefrontal": ("ORB", "PL", "ILA", "ACA", "FRP"),
    "lateral": ("AI", "GU", "VISC", "TEa", "PERI", "ECT"),
}


class Dataset:
    """Selective neurons, their encoding coefficients, and the region metadata.

    d = Dataset()
    d.regions              regions with enough selective neurons, ordered by hierarchy
    d.features("temporal") one row per neuron: the features that define its tuning
    d.neurons("VISp")      indices of the selective neurons of a region
    d.connectivity()       K x K log anatomical connectivity between d.regions
    d.hierarchy()          position of each region along the cortical hierarchy
    """

    def __init__(self, npz=None, min_neurons=MIN_NEURONS):
        if npz is None:
            npz = paths.external("posani2026", "rrr_neurons.npz")
        z = np.load(npz, allow_pickle=True)
        self.acronym, self.dR2, self.beta = z["acronym"], z["dR2"], z["beta"]
        self.n_vars, self.n_time = self.beta.shape[1], self.beta.shape[2]

        # cortical areas, ordered from sensory to associative
        self.area_list = pd.read_csv(
            paths.derived("posani2026", "area_list.csv"), header=None).values[:, 0]
        self._conn = np.log(pd.read_csv(
            paths.derived("harris2019", "ctx2ctx_conn.csv"), header=None).values + 1e-8)
        self._area2i = {a: i for i, a in enumerate(self.area_list)}

        self.keep = (self.dR2 > DR2_THRESHOLD) & np.isin(self.acronym, self.area_list)
        self.regions = np.array([a for a in self.area_list
                                 if len(self.neurons(a)) >= min_neurons])
        self.counts = np.array([len(self.neurons(a)) for a in self.regions])

    def neurons(self, region):
        return np.flatnonzero(self.keep & (self.acronym == region))

    def features(self, mode="temporal"):
        """Per-neuron tuning features, i.e. the 'conditions' of the shape analysis.

        "selectivity" : alpha_v = sum_t |beta_vt|, the 8 values averaged in their Fig. 2c
        "temporal"    : the full time-varying coefficients, 8 variables x 100 time bins
        """
        if mode == "selectivity":
            return np.abs(self.beta).sum(2)
        if mode == "temporal":
            return self.beta.reshape(len(self.beta), self.n_vars * self.n_time)
        raise ValueError(mode)

    def connectivity(self):
        """Symmetrised log anatomical connectivity between the selected regions."""
        i = [self._area2i[a] for a in self.regions]
        C = self._conn[np.ix_(i, i)]
        return (C + C.T) / 2

    def hierarchy(self):
        return np.array([self._area2i[a] for a in self.regions])

    def module(self, region):
        for m, prefixes in MODULES.items():
            if any(region.startswith(p) for p in prefixes):
                return m
        return "other"

    def modules(self):
        return np.array([self.module(a) for a in self.regions])

    def __repr__(self):
        return (f"Dataset({self.keep.sum()} selective neurons, "
                f"{len(self.regions)} regions)")
