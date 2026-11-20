"""Small, deterministic helpers shared by preparation, fit and application."""

from __future__ import annotations

import sys

from pathlib import Path

import numpy as np

import config

SCRIPTS_DIR = Path(__file__).resolve().parents[1]

CODE_DIR = Path(__file__).resolve().parent

PROXY = "/afs/cern.ch/user/j/jiehan/x509up_u152815"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def bjet_veto_mask(arrays):
    medium, loose = (
        np.asarray(arrays[name], dtype=float)
        for name in ("nbjets_medium", "nbjets_loose")
    )
    for value in (medium, loose):
        if (
            not np.isfinite(value).all()
            or np.any(value < 0)
            or np.any(value != np.floor(value))
        ):
            raise ValueError("Invalid stored b-tag multiplicity")
    return (medium == 0) & (loose <= 1)


def normalization_sumw(root_file, adapter):
    branch = {"native": "genEventSumw", "sampled": "normalization_sumw"}[adapter]
    values = np.asarray(
        root_file["conditions"][branch].array(library="np"), dtype=float
    )
    total = float(values.sum())
    if not np.isfinite(values).all() or not np.isfinite(total) or total == 0:
        raise ValueError(f"Invalid conditions/{branch}: {total}")
    return total


def selected_jet_mask(pt, eta):
    pt = np.asarray(pt, dtype=float)
    eta = np.asarray(eta, dtype=float)
    keep = np.isfinite(pt) & np.isfinite(eta) & (pt > 25.0) & (np.abs(eta) < 4.7)
    return keep


def bin_indices(values, edges):
    values = np.asarray(values, dtype=float)
    edges = np.asarray(edges, dtype=float)
    result = np.searchsorted(edges, values, side="right") - 1
    valid = np.isfinite(values) & (values >= edges[0]) & (values < edges[-1])
    return result, valid


def jet_indices(pt, eta, matched):
    pt_idx, pt_valid = bin_indices(pt, config.JET_PT_EDGES)
    eta_idx, eta_valid = bin_indices(eta, config.ETA_EDGES)
    valid = pt_valid & eta_valid & selected_jet_mask(pt, eta)
    matched = np.asarray(matched, dtype=np.int32)
    if np.any((matched < 0) | (matched > 1)):
        raise ValueError("Jet_isGenMatched must contain only zero and one")
    cell = (
        (1 - matched) * (config.N_PT * config.N_ETA) + pt_idx * config.N_ETA + eta_idx
    )
    return cell.astype(np.int32), valid


def mc_jet_veto(cells, keep, is_data):
    if not is_data and config.MC_VETO_CELLS:
        keep = keep & ~np.isin(cells, config.MC_VETO_CELLS)
    return keep


def selected_jets(arrays, is_data, *, apply_veto=True, statistics=None):
    """Keep each event and its leading 32 selected jets, with aligned jet fields."""
    import awkward as ak

    values = [
        arrays[name]
        for name in ("Jet_corrPt", "Jet_corrEta", "Jet_corrPhi", "Jet_corrMass")
    ]
    counts = ak.to_numpy(ak.num(values[0]))
    labels = ak.zeros_like(values[0]) if is_data else arrays["Jet_isGenMatched"]
    for value in (*values[1:], labels):
        if not np.array_equal(counts, ak.to_numpy(ak.num(value))):
            raise ValueError("Jet branch length mismatch")
    cells, keep = jet_indices(
        *(ak.to_numpy(ak.flatten(value)) for value in (values[0], values[1], labels))
    )
    if apply_veto:
        keep = mc_jet_veto(cells, keep, is_data)
    mask = ak.unflatten(keep, counts)
    selected = tuple(value[mask] for value in values) + (
        ak.unflatten(cells, counts)[mask],
    )
    original = ak.to_numpy(ak.num(selected[0]))
    if statistics is not None:
        statistics["truncated_events"] = statistics.get("truncated_events", 0) + int(
            np.count_nonzero(original > config.MAX_JETS)
        )
        statistics["discarded_jets"] = statistics.get("discarded_jets", 0) + int(
            np.maximum(original - config.MAX_JETS, 0).sum()
        )
        statistics["max_selected_jets"] = max(
            statistics.get("max_selected_jets", 0), int(original.max(initial=0))
        )
    order = ak.argsort(selected[0], axis=1, ascending=False, stable=True)[
        :, : config.MAX_JETS
    ]
    return tuple(value[order] for value in selected)


def four_vector(pt, eta, phi, mass):
    px, py = pt * np.cos(phi), pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(np.maximum(px * px + py * py + pz * pz + mass * mass, 0.0))
    return px, py, pz, energy


def dilepton_kinematics(arrays, region="zcr"):
    if region == "zee":
        return tuple(
            np.asarray(arrays[name], dtype=float)
            for name in ("zee_pt", "zee_phi", "zee_mass")
        )
    vectors = np.zeros((4, len(arrays)), dtype=float)
    for i in (1, 2):
        pt = np.asarray(arrays[f"pt_kit_bsc_{i}"], dtype=float)
        eta = np.asarray(arrays[f"mu{i}_fromH_eta"], dtype=float)
        phi = np.asarray(arrays[f"mu{i}_fromH_phi"], dtype=float)
        vectors += np.asarray(four_vector(pt, eta, phi, np.full(len(pt), 0.1056584)))
        photon_pt = np.asarray(arrays[f"fsrPhoton_pt_{i}"], dtype=float)
        keep = (np.asarray(arrays[f"fsrPhoton_dROverEt2_{i}"], dtype=float) < 0.012) & (
            photon_pt > 0
        )
        vectors += np.asarray(
            four_vector(
                np.where(keep, photon_pt, 0.0),
                np.where(
                    keep, np.asarray(arrays[f"fsrPhoton_eta_{i}"], dtype=float), 0.0
                ),
                np.where(
                    keep, np.asarray(arrays[f"fsrPhoton_phi_{i}"], dtype=float), 0.0
                ),
                np.zeros(len(pt)),
            )
        )
    px, py, pz, energy = vectors
    pt = np.hypot(px, py)
    mass = np.sqrt(np.maximum(energy * energy - px * px - py * py - pz * pz, 0.0))
    phi = np.arctan2(py, px)
    return pt, phi, mass


def nominal_weight_scale(root_file, path, region, is_data):
    if is_data:
        return 1.0
    from samples import input_record

    record = input_record(path, region)
    xsec = record["cross_section_pb"]
    if xsec is None or not np.isfinite(xsec) or xsec <= 0:
        raise ValueError(f"Invalid cross section for {path}")
    sumw = normalization_sumw(root_file, config.ERAS[config.ERA]["adapter"])
    return config.LUMI_PB * xsec / sumw


def event_weights(arrays, scale, region, is_data):
    if is_data:
        return np.ones(len(arrays), dtype=np.float64)
    names = config.weight_branches(region)
    result = np.full(len(arrays), scale, dtype=np.float64)
    for name in names:
        result *= np.asarray(arrays[name], dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite nominal event weights")
    return result


def selection(arrays, region):
    mask = bjet_veto_mask(arrays)
    if region != "zee":
        nmu = np.asarray(arrays["nmuons"], dtype=int)
        nele = np.asarray(arrays["nelectrons"], dtype=int)
    if region in ("zcr", "srsb"):
        ptll, _, mass = dilepton_kinematics(arrays)
        mask &= np.asarray(arrays["trg_single_mu24"], dtype=bool)
        mask &= (nmu + nele) < 3
        mass_window = (
            (mass > 70) & (mass < 110)
            if region == "zcr"
            else ((mass > 110) & (mass < 115)) | ((mass > 135) & (mass < 150))
        )
        mask &= np.isfinite(mass) & mass_window & np.isfinite(ptll)
    elif region == "zee":
        for name in (
            "zee_pair_flag",
            "zee_os_flag",
            "zee_extra_veto_flag",
            "trg_single_ele30",
        ):
            mask &= np.asarray(arrays[name]) == 1
        ptll, phi, mass = dilepton_kinematics(arrays, region)
        mask &= (
            np.isfinite(ptll)
            & np.isfinite(phi)
            & np.isfinite(mass)
            & (mass > 70)
            & (mass < 110)
        )
    elif region == "wcr":
        mask &= np.asarray(arrays["single_muon_trigger_flag"], dtype=bool)
        mask &= (nmu == 1) & (nele == 0)
        met = np.asarray(arrays["met_pt"], dtype=float)
        mt = np.asarray(arrays["muon_met_mt"], dtype=float)
        mask &= np.isfinite(met) & (met > 30) & np.isfinite(mt) & (mt > 50)
    else:
        raise ValueError(f"Unsupported extraction region: {region}")
    return mask
