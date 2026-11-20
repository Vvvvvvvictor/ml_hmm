"""Validate prefit and corrected distributions in the reserved dimuon samples."""

from __future__ import annotations

import awkward as ak

import numpy as np

from scipy import sparse

from histogram_accumulation import one_hot_counts

from config import DY_PT_EDGES

from until import jet_indices, mc_jet_veto

import json
from pathlib import Path

_VARIABLES = json.loads(
    (Path(__file__).resolve().parents[2] / "variables.json").read_text()
)
HISTOGRAMS = {
    k: (np.asarray(v["edges"], dtype=float), v["label"])
    for k, v in _VARIABLES["base"].items()
}

STAGES = ("nominal", "jet_sf", "dy_theory", "jet_sf_dy_theory")

NORMALIZATION_SCHEMA = 6


def _jets(arrays, is_data, apply_veto=True, statistics=None):
    from until import selected_jets

    return selected_jets(arrays, is_data, apply_veto=apply_veto, statistics=statistics)


def _jet_counts(values, edges):
    flat = ak.to_numpy(ak.flatten(values))
    indices = np.searchsorted(edges, flat, side="right") - 1
    rows = np.repeat(np.arange(len(values)), ak.to_numpy(ak.num(values)))
    valid = np.isfinite(flat) & (indices >= 0) & (indices < len(edges) - 1)
    return sparse.csr_matrix(
        (
            np.ones(np.count_nonzero(valid), dtype=np.int32),
            (rows[valid], indices[valid]),
        ),
        shape=(len(values), len(edges) - 1),
    )


def _observable_counts(jets, ptll=None):
    pt, eta, phi, mass, _ = jets
    njet = ak.to_numpy(ak.num(pt))
    result = {
        name: _jet_counts(values, HISTOGRAMS[name][0])
        for name, values in (("jet_pt", pt), ("jet_eta", eta))
    }
    target = (eta > 2.5) & (eta < 2.8) & (pt < 50)
    target_cells = (
        np.searchsorted(
            np.asarray([25, 30, 35, 40, 50]),
            ak.to_numpy(ak.flatten(pt[target])),
            side="right",
        )
        - 1
    ) * 3 + (
        np.searchsorted(
            np.asarray([2.5, 2.6, 2.7, 2.8]),
            ak.to_numpy(ak.flatten(eta[target])),
            side="right",
        )
        - 1
    )
    result["forward_lowpt_cell"] = _jet_counts(
        ak.unflatten(target_cells, ak.num(pt[target])),
        HISTOGRAMS["forward_lowpt_cell"][0],
    )
    order = ak.argsort(pt, ascending=False)[:, :2]
    p, e, ph, m = (values[order] for values in (pt, eta, phi, mass))
    px, py, pz = (
        ak.to_numpy(ak.sum(v, axis=1))
        for v in (p * np.cos(ph), p * np.sin(ph), p * np.sinh(e))
    )
    energy = ak.to_numpy(ak.sum(np.sqrt(p**2 * np.cosh(e) ** 2 + m**2), axis=1))
    mjj = np.sqrt(np.maximum(energy**2 - px**2 - py**2 - pz**2, 0))
    e2, p2 = ak.pad_none(e, 2, clip=True), ak.pad_none(p, 2, clip=True)
    deta = ak.to_numpy(ak.fill_none(abs(e2[:, 0] - e2[:, 1]), np.nan))
    vbf = ak.to_numpy(
        ak.fill_none(
            (p2[:, 0] > 35)
            & (p2[:, 1] > 25)
            & (abs(e2[:, 0]) < 2.4)
            & (abs(e2[:, 1]) < 2.4),
            False,
        )
    )
    vbf &= (deta >= 2.5) & (mjj >= 400)
    values = {
        "njet": njet,
        "delta_eta_jj": deta,
        "mjj": np.where(njet >= 2, mjj, np.nan),
        "vbf_mjj": np.where(vbf, mjj, np.nan),
    }
    if ptll is not None:
        values["ptll"] = ptll
    for name, value in values.items():
        edges = HISTOGRAMS[name][0]
        indices = np.searchsorted(edges, value, side="right") - 1
        valid = np.isfinite(value) & (indices >= 0) & (indices < len(edges) - 1)
        result[name] = one_hot_counts(indices, len(edges) - 1, valid)
    return result
