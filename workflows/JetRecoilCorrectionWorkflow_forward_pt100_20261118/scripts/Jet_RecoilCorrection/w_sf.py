"""Positive W+jets scale factors indexed by selected reconstructed jet count."""

import numpy as np

W_SF_DEFINITION = dict(
    w_sf_mode="reco_njet_mumet_pt",
    w_njet_edges=[0, 2, 3, 4, "inf"],
    w_njet_labels=["0–1", "2", "3", "≥4"],
    w_sf_applied_regions=["wcr"],
    w_sf_catalog_group="W",
    shared_w_sf_across_eras=False,
    wcr_observation_blocks=["jet_pt_eta", "reco_njet_01_2_3_4p"],
    wcr_covariance="full_event_level_including_jet_njet_cross_terms",
)


def w_bins(njet):
    return np.searchsorted([2, 3, 4], np.asarray(njet), side="right")


W_PT_EDGES = [0.0, 30.0, 70.0, 150.0, np.inf]

MUON_PT_EDGES = [0.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 80.0, 120.0, 200.0, np.inf]

W_CLOSURE_EDGES = dict(
    muon_pt=MUON_PT_EDGES,
    mt=[50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 120.0, 160.0, 240.0, np.inf],
    met=[30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 150.0, 250.0, np.inf],
    recoil=[0.0, 10.0, 20.0, 30.0, 50.0, 70.0, 100.0, 150.0, 250.0, np.inf],
)

W_PLOT_EDGES = {
    name: [*range(low, 101, 5), 120.0, 160.0, 200.0, 250.0, np.inf]
    for name, low in (("muon_pt", 0), ("mt", 50), ("met", 30), ("recoil", 0))
}


from config import DY_PT_EDGES

W_MUMET_PT_EDGES = np.asarray(DY_PT_EDGES, dtype=float)

N_W_SF = 4 * (len(W_MUMET_PT_EDGES) - 1)

W_MUMET_COLUMN = 116


def mumet_pt(arrays):
    import awkward as ak

    mu, phi, met, mphi = [
        ak.to_numpy(arrays[name]).astype(float)
        for name in ("single_muon_pt", "single_muon_phi", "met_pt", "met_phi")
    ]
    values = np.hypot(
        mu * np.cos(phi) + met * np.cos(mphi), mu * np.sin(phi) + met * np.sin(mphi)
    )
    if np.any(~np.isfinite(values) | (values < 0)):
        raise ValueError("Invalid muon+MET transverse momentum")
    return values


def mumet_bins(pt):
    pt = np.asarray(pt, dtype=float)
    if np.any(~np.isfinite(pt) | (pt < 0)):
        raise ValueError("Invalid muon+MET transverse momentum")
    return np.searchsorted(W_MUMET_PT_EDGES[1:-1], pt, side="right")


def w_sf_indices(njet, pt_bin):
    pt_bin = np.asarray(pt_bin, dtype=int)
    if np.any((pt_bin < 0) | (pt_bin >= N_W_SF // 4)):
        raise ValueError("Missing or invalid fine muon+MET pT bin")
    return pt_bin * 4 + w_bins(njet)


W_SF_DEFINITION.update(
    w_mumet_pt_edges=[*map(float, W_MUMET_PT_EDGES[:-1]), "inf"],
    w_sf_shape=[N_W_SF // 4, 4],
    w_sf_index="pt_bin * 4 + njet_bin",
    wcr_observation_blocks=[
        "jet_pt_eta",
        "reco_njet_01_2_3_4p",
        "muon_pt",
        "mumet_pt_njet",
    ],
    wcr_covariance="full_event_level_including_all_observation_cross_terms",
)
