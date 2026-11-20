from runtime import *

from w_sf import W_CLOSURE_EDGES

import numpy as np


def reco_values(a, *, edges=W_CLOSURE_EDGES):
    import awkward as ak

    mu = ak.to_numpy(a.single_muon_pt).astype(float)
    phi = ak.to_numpy(a.single_muon_phi).astype(float)
    met = ak.to_numpy(a.met_pt).astype(float)
    mphi = ak.to_numpy(a.met_phi).astype(float)
    recoil = np.hypot(
        mu * np.cos(phi) + met * np.cos(mphi), mu * np.sin(phi) + met * np.sin(mphi)
    )
    values = [mu, ak.to_numpy(a.muon_met_mt), met, recoil]
    if any(np.any(~np.isfinite(v) | (v < 0)) for v in values):
        raise ValueError("Invalid WCR reco observable")
    return np.column_stack(
        [
            np.searchsorted(edges[name][1:-1], v, side="right")
            for name, v in zip(edges, values)
        ]
    ).astype(np.uint16)
