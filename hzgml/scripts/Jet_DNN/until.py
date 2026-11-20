"""Shared configuration and small numerical helpers for Jet DNN."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import awkward as ak
import numpy as np


ERA = os.environ.get("JET_DNN_ERA", "2022EE")
DEFAULT_WORK = Path("/eos/user/j/jiehan/HMuMuShare/JetDNN_2022_nohehf")
WORK = Path(os.environ.get(
    "JET_DNN_WORK_ROOT",
    str(DEFAULT_WORK.parent / "JetDNN_2022EE_nohehf" if ERA == "2022EE" else DEFAULT_WORK),
))
FIT_INPUT = WORK / f"feature_variants/{os.environ.get('JET_DNN_FEATURE_SET', 'no_rho')}/fit_input"
FIT_OUTPUT = WORK / "fit_results"
VALIDATION_OUTPUT = WORK / "validation"
WCR_INPUT = Path(f"/eos/user/j/jiehan/HMuMuShare/WCR_{ERA}_nohehf" if ERA == "2022EE"
                 else "/eos/user/j/jiehan/HMuMuShare/WCR_2022")
WCR_OUTPUT = VALIDATION_OUTPUT / "wcr"
EVENT_NORMALIZATION = WORK / "event_norm_refit_bveto_v2/event_norm.json"
ZCR_INPUT = Path(f"/eos/user/j/jiehan/HMuMuShare/ZCR_{ERA}_nohehf" if ERA == "2022EE"
                 else "/eos/user/j/jiehan/HMuMuShare/ZCR_2022_nohehf")
SRSB_INPUT = Path(f"/eos/user/j/jiehan/HMuMuShare/SRSB_{ERA}_nohehf" if ERA == "2022EE"
                  else "/eos/user/j/jiehan/HMuMuShare/SRSB_2022_nohehf")
LUMI = 26674.924045014 if ERA == "2022EE" else 7989.513665822
SCORE_SEED = 20250716
THINNING_SEED = 12345
LAMBDA_M = 10.0
LAMBDA_U = 10.0
FIT_PT_BINS = (20, 22.5, 25, 27.5, 30, 34.5, 40, 45, 50, 60, 90, np.inf)
FIT_HF_ETA_BOUNDARY = float(os.environ.get("JET_DNN_HF_ETA_BOUNDARY", "2.964"))
FIT_SCORE_EDGES = tuple(map(float, json.loads(os.environ.get(
    "JET_DNN_FIT_SCORE_EDGES", "[0, 0.35, 0.45, 0.55, 0.65, 1]"
))))
FIT_ETA_FINE_BINS = (
    -4.716, -4.538, -4.363, -4.191, -4.013, -3.839, -3.664, -3.489,
    -3.314, -3.197, -3.081, -FIT_HF_ETA_BOUNDARY, -2.853, -2.650, -2.500, -2.322, -2.172,
    -2.043, -1.566, -1.305, -0.879, 0.0, 0.879, 1.305, 1.566,
    2.043, 2.172, 2.322, 2.500, 2.650, 2.853, FIT_HF_ETA_BOUNDARY, 3.081, 3.197,
    3.314, 3.489, 3.664, 3.839, 4.013, 4.191, 4.363, 4.538, 4.716,
)
FIT_CELL_COUNT = 220
FIT_TEMPLATE_STAT_NUISANCE = True
B_VETO_BRANCHES = ("nbjets_medium", "nbjets_loose")
B_VETO_DESCRIPTION = "(nbjets_medium < 1) & (nbjets_loose < 2)"
AGREEMENT_PLOT_VERSION = 3
FIT_ALIASES = {
    "DYto2L_2Jets_MLL50.root": "DYto2L_2Jets_MLL50.root",
    "EWK_2L2J_Herwig.root": "EWK_2L2J_Herwig.root",
    "TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8.root": "TTto2L2Nu.root",
    "data.root": "data.root",
}

XSECTIONS = {
    "DYto2Mu-2Jets_Bin-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8": 2091.7,
    "DYto2Mu-2Jets_Bin-MLL-105to160_TuneCP5_13p6TeV_amcatnloFXFX-pythia8": 48.196713,
    "EWK-2L2J_Bin-MLL-50-MJJ-120_TuneCH3_13p6TeV_madgraph-herwig7": 1.2013304,
    "EWK-2Mu2J_Bin-M2Mu-105to160-M2J-120_TuneCP5_13p6TeV_madgraph-pythia8": 0.091062946,
    "EWK-2Mu2J_Bin-MLL-105to160_TuneCH3_13p6TeV_madgraph-herwig7": 0.091062946,
    "GluGluH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8": 0.0114359,
    "VBFH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8": 0.000884275,
    "TTH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powheg-pythia8": 0.00012397720,
    "WminusH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8": 0.00012352368,
    "WplusH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8": 0.00019717096,
    "ZH-Hto2Mu_Par-M-125_TuneCP5_13p6TeV_powhegMINLO-pythia8": 0.00020535717,
    "WWW-4F_TuneCP5_13p6TeV_amcatnlo-pythia8": 0.2328,
    "WWZ-4F_TuneCP5_13p6TeV_amcatnlo-pythia8": 0.1851,
    "WZZ-5F_TuneCP5_13p6TeV_amcatnlo-pythia8": 0.06206,
    "ZZZ-5F_TuneCP5_13p6TeV_amcatnlo-pythia8": 0.01591,
    "DYto2L_2Jets_MLL50": 6345.99,
    "DYto2Mu_2Jets_MLL105To160": 48.196713,
    "EWK_2L2J_Herwig": 1.2013304,
    "EWK_2Mu2J_Bin-MLL-105to160_TuneCH3_13p6TeV_madgraph-herwig7": 0.091062946,
    "EWK-2Mu2J_M2Mu-105to160_M2J-120_TuneCP5_13p6TeV_madgraph-pythia8": 0.091062946,
    "TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8": 97.4488,
    "TWminusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8": 4.6511,
    "TbarWplusto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8": 4.6511,
    "TZQB-4FS_OnshellZ_TuneCP5_13p6TeV_amcatnlo-pythia8": 0.0801,
    "TZQB-Zto2L-4FS_MLL-30_TuneCP5_13p6TeV_amcatnlo-pythia8": 0.0801,
    "WWW_4F_TuneCP5_13p6TeV_amcatnlo-madspin-pythia8": 0.2328,
    "WWZ_4F_TuneCP5_13p6TeV_amcatnlo-pythia8": 0.1851,
    "WZZ_TuneCP5_13p6TeV_amcatnlo-pythia8": 0.06206,
    "ZZZ_TuneCP5_13p6TeV_amcatnlo-pythia8": 0.01591,
    "WWto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8": 12.98,
    "WZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8": 8.17,
    "WZto3LNu_TuneCP5_13p6TeV_powheg-pythia8": 5.31,
    "ZZto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8": 1.19,
    "ZZto2L2Q_TuneCP5_13p6TeV_powheg-pythia8": 8.08,
    "ZZto4L_TuneCP5_13p6TeV_powheg-pythia8": 1.65,
    "GluGluHto2Mu": 0.0114359,
    "VBFHto2Mu": 0.000884275,
    "TTH_Hto2Mu": 0.00012397720,
    "WminusH_Hto2Mu_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8": 0.00012352368,
    "WplusH_Hto2Mu_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8": 0.00019717096,
    "ZH_Hto2Mu_M-125_TuneCP5_13p6TeV_powheg-minlo-pythia8": 0.00020535717,
}

WCR_BRANCHING = {"tt_lep": 0.3258, "tt_had": 0.6742}
WCR_XSECTIONS = {
    "WtoMuNu-2Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.root": 20850.9,
    "DYto2Mu-2Jets_Bin-MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.root": 2091.7,
    "WtoLNu-2Jets_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.root": 62552.7,
    "WtoLNu-2Jets-EWK_TuneCP5_13p6TeV_madgraph-pythia8.root": 41.02,
    "DYto2L-2Jets_MLL-10to50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.root": 19317.5,
    "DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.root": 6244.8,
    "DYto2L-2Jets-EWK_MLL-50_TuneCP5_13p6TeV_madgraph-pythia8.root": 7.101,
    "TTto2L2Nu_TuneCP5_13p6TeV_powheg-pythia8.root": 921.0 * WCR_BRANCHING["tt_lep"] ** 2,
    "TTtoLNu2Q_TuneCP5_13p6TeV_powheg-pythia8.root": 921.0 * 2.0 * WCR_BRANCHING["tt_lep"] * WCR_BRANCHING["tt_had"],
}
WCR_SAMPLES = ("data", *(Path(name).stem for name in WCR_XSECTIONS))


def raw_files(base):
    root = Path(base)
    data = sorted(root.glob("data.root")) or sorted((root / "data").glob("*.root"))
    return data + sorted((root / "mc").glob("*.root"))


def sample_stem(path):
    return Path(path).stem


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    os.close(descriptor)
    try:
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def correction_edges(edges):
    return ["inf" if np.isposinf(value) else float(value) for value in edges]


def correction_binning(input_name, edges, content, flow="clamp"):
    return {
        "nodetype": "binning",
        "input": input_name,
        "edges": correction_edges(edges),
        "content": content,
        "flow": flow,
    }


def correction_component(sf_map, pt_bins, eta_bins):
    content = []
    for eta_index in range(len(eta_bins) - 1):
        values = [
            float(sf_map[pt_index, eta_index])
            if np.isfinite(sf_map[pt_index, eta_index]) and sf_map[pt_index, eta_index] >= 0.0
            else 1.0
            for pt_index in range(len(pt_bins) - 1)
        ]
        content.append(correction_binning("pt", pt_bins, values))
    return correction_binning("eta", eta_bins, content)


def correction_set(matched_map, unmatched_map, pt_bins=FIT_PT_BINS, eta_bins=FIT_ETA_FINE_BINS):
    eta_bins = np.asarray(eta_bins, dtype=float)
    if eta_bins.ndim != 1 or len(eta_bins) < 2 or np.any(np.diff(eta_bins) <= 0.0):
        raise ValueError("Correction eta edges must be strictly increasing")
    inputs = [
        {"name": "pt", "type": "real", "description": "Jet corrected transverse momentum [GeV]"},
        {"name": "eta", "type": "real", "description": "Signed jet corrected pseudorapidity"},
    ]
    output = {"name": "sf", "type": "real"}
    return {
        "schema_version": 2,
        "description": "Jet gen-match scale factors from a score-template fit",
        "corrections": [
            {
                "name": "jet_genmatched_sf",
                "version": 3,
                "description": "Scale factor for gen-matched jets",
                "inputs": inputs,
                "output": output,
                "data": correction_component(matched_map, pt_bins, eta_bins),
            },
            {
                "name": "jet_unmatched_sf",
                "version": 3,
                "description": "Scale factor for gen-unmatched jets",
                "inputs": inputs,
                "output": output,
                "data": correction_component(unmatched_map, pt_bins, eta_bins),
            },
        ],
    }


def unity_correction_set(pt_bins=FIT_PT_BINS, eta_bins=FIT_ETA_FINE_BINS):
    eta_bins = np.asarray(eta_bins, dtype=float)
    if eta_bins.ndim != 1 or len(eta_bins) < 2 or np.any(np.diff(eta_bins) <= 0.0):
        raise ValueError("Correction eta edges must be strictly increasing")
    values = np.ones((len(pt_bins) - 1, len(eta_bins) - 1), dtype=float)
    payload = correction_set(values, values, pt_bins, eta_bins)
    payload["description"] = "Unity jet correction for nominal comparisons"
    return payload


def read_tree(path, expressions=None, *, tree="ntuple", entry_stop=None, library="ak"):
    import uproot

    with uproot.open(path, object_cache=None, array_cache=None) as root_file:
        return root_file[tree].arrays(
            expressions, entry_stop=entry_stop, library=library
        )


def gen_event_sumw_values(root_file):
    if "conditions" not in root_file or "genEventSumw" not in root_file["conditions"]:
        raise KeyError("ROOT file has no conditions/genEventSumw")
    branch = "normalization_sumw" if "normalization_sumw" in root_file["conditions"] else "genEventSumw"
    values = np.asarray(root_file["conditions"][branch].array(library="np"))
    if not np.isfinite(values.astype(float)).all():
        raise ValueError("conditions/genEventSumw contains non-finite values")
    return values


def randoms(seed, run, event, path, index):
    values = np.empty(len(index), dtype=float)
    for position, (run_value, event_value, jet_index) in enumerate(zip(run, event, index)):
        payload = f"{seed}:{int(run_value)}:{int(event_value)}:{path}:{int(jet_index)}".encode()
        values[position] = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") / float(1 << 64)
    return values


def flat(value):
    return ak.to_numpy(ak.flatten(value, axis=1))


def thinning_region_mask(active, pt, eta, eta_min=2.5):
    return np.asarray(active, dtype=bool) & (np.abs(np.asarray(eta, dtype=float)) > eta_min)


def four_vector(pt, eta, phi, mass):
    px, py = pt * np.cos(phi), pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(np.maximum(px * px + py * py + pz * pz + mass * mass, 0.0))
    return px, py, pz, energy


def scalar_values(pt, eta, phi, mass, h_eta, h_phi, min_pt=25.0):
    order = ak.argsort(pt, axis=1, ascending=False)
    ordered = [value[order] for value in (pt, eta, phi, mass)]
    padded = [ak.fill_none(ak.pad_none(value, 2, axis=1), -100.0) for value in ordered]
    p1, e1, f1, m1 = [np.asarray(value[:, 0], dtype=float) for value in padded]
    p2, e2, f2, m2 = [np.asarray(value[:, 1], dtype=float) for value in padded]
    valid1, valid2 = np.isfinite(p1) & (p1 > min_pt), np.isfinite(p2) & (p2 > min_pt)
    njet = np.asarray(ak.sum(pt > min_pt, axis=1), dtype=np.int32)
    p1, e1, f1, m1 = [np.where(valid1, value, -100.0) for value in (p1, e1, f1, m1)]
    p2, e2, f2, m2 = [np.where(valid2, value, -100.0) for value in (p2, e2, f2, m2)]
    px1, py1, pz1, en1 = four_vector(np.maximum(p1, 0), np.where(valid1, e1, 0), np.where(valid1, f1, 0), np.maximum(m1, 0))
    px2, py2, pz2, en2 = four_vector(np.maximum(p2, 0), np.where(valid2, e2, 0), np.where(valid2, f2, 0), np.maximum(m2, 0))
    px, py, pz, energy = px1 + px2, py1 + py2, pz1 + pz2, en1 + en2
    dijet_pt = np.hypot(px, py)
    dijet_mass = np.sqrt(np.maximum(energy * energy - px * px - py * py - pz * pz, 0.0))
    dijet_eta = np.arcsinh(np.divide(pz, dijet_pt, out=np.zeros_like(pz), where=dijet_pt > 0))
    dijet_phi = np.arctan2(py, px)
    both = valid1 & valid2
    dijet_mass, dijet_eta = np.where(both, dijet_mass, -100.0), np.where(both, dijet_eta, -100.0)
    dijet_pt, dijet_phi = np.where(both, dijet_pt, -100.0), np.where(both, dijet_phi, -100.0)
    dphi = np.abs(f1 - f2)
    dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
    deta = np.abs(e1 - e2)
    dphi_z1 = np.abs(h_phi - f1)
    dphi_z1 = np.where(dphi_z1 > np.pi, 2 * np.pi - dphi_z1, dphi_z1)
    dphi_z2 = np.abs(h_phi - f2)
    dphi_z2 = np.where(dphi_z2 > np.pi, 2 * np.pi - dphi_z2, dphi_z2)
    min_deta = np.minimum(np.abs(h_eta - e1), np.abs(h_eta - e2))
    min_dphi = np.minimum(dphi_z1, dphi_z2)
    for value in (deta, dphi, min_deta, min_dphi):
        value[~both] = -100.0
    min_deta[valid1 & ~valid2] = np.abs(h_eta[valid1 & ~valid2] - e1[valid1 & ~valid2])
    min_dphi[valid1 & ~valid2] = dphi_z1[valid1 & ~valid2]
    return {
        "njets": njet, "jet1_pt": p1, "jet1_eta": e1, "jet1_phi": f1, "jet1_mass": m1,
        "jet2_pt": p2, "jet2_eta": e2, "jet2_phi": f2, "jet2_mass": m2,
        "dijet_mass": dijet_mass, "dijet_eta": dijet_eta, "dijet_pt": dijet_pt, "dijet_phi": dijet_phi,
        "delta_eta_jj": deta, "delta_phi_jj": dphi,
        "min_delta_eta_dimu_jets": min_deta, "min_delta_phi_dimu_jets": min_dphi,
    }


def jme_keep_mask(pt, eta, *, min_pt=25.0, forward_pt=50.0, eta_boundary=2.5):
    finite = ak.is_valid(pt) & ak.is_valid(eta)
    abs_eta = np.abs(eta)
    return finite & (pt > min_pt) & (abs_eta < 4.7) & ((abs_eta <= eta_boundary) | (pt >= forward_pt))


def thinning_candidates(active, labels, sf, region):
    labels, sf = np.asarray(labels, dtype=np.int8), np.asarray(sf, dtype=float)
    active, region = np.asarray(active, dtype=bool), np.asarray(region, dtype=bool)
    if len(region) != len(active):
        raise ValueError("Thinning-region mask length mismatch")
    return active & region & np.isin(labels, (0, 1)) & (sf < 1.0)


def aggregate_event_sf(factors, counts, positive_mask, denominator_mask):
    factors = np.asarray(factors, dtype=float)
    counts = np.asarray(counts, dtype=np.int64)
    positive_mask, denominator_mask = np.asarray(positive_mask, dtype=bool), np.asarray(denominator_mask, dtype=bool)
    if len(positive_mask) != len(factors) or len(denominator_mask) != len(factors):
        raise ValueError("Event-SF mask length mismatch")
    jagged_factors = ak.unflatten(factors, counts)
    jagged_mask = ak.unflatten(positive_mask, counts)
    jagged_denominator = ak.unflatten(denominator_mask, counts)
    selected = ak.to_numpy(ak.flatten(ak.where(jagged_mask, jagged_factors, 1.0)))
    if np.any(~np.isfinite(selected)) or np.any(selected <= 0.0):
        raise ValueError("Positive-root event-SF requires finite positive factors")
    log_sum = ak.to_numpy(ak.sum(ak.where(jagged_mask, np.log(jagged_factors), 0.0), axis=1))
    denominator = ak.to_numpy(ak.sum(jagged_denominator, axis=1))
    return np.where(denominator > 0, np.exp(log_sum / np.maximum(denominator, 1)), 1.0)


def bjet_veto_mask(arrays):
    fields = getattr(arrays, "fields", None)
    available = set(fields) if fields is not None else set(arrays)
    missing = [name for name in B_VETO_BRANCHES if name not in available]
    if missing:
        raise KeyError(f"Missing b-jet veto branches: {missing}")
    values = [np.asarray(arrays[name], dtype=float) for name in B_VETO_BRANCHES]
    for name, value in zip(B_VETO_BRANCHES, values):
        if not np.isfinite(value).all() or np.any(value < 0) or np.any(value != np.floor(value)):
            raise ValueError(f"Invalid {name} values for b-jet veto")
    return (values[0] < 1) & (values[1] < 2)


def compare_histograms(old, new, normalization):
    data = np.asarray(old["data_sumw"], dtype=float)
    data2 = np.asarray(old["data_sumw2"], dtype=float)
    results, supports = {}, {}
    for key, value in (("old", old), ("new", new)):
        model_data = np.asarray(value["data_sumw"], dtype=float)
        model_data2 = np.asarray(value["data_sumw2"], dtype=float)
        mc = np.asarray(value["mc_sumw"], dtype=float)
        mc2 = np.asarray(value["mc_sumw2"], dtype=float)
        if any(array.shape != data.shape for array in (model_data, model_data2, mc, mc2)):
            raise ValueError("Histogram arrays have inconsistent lengths")
        scale = float(model_data.sum() / mc.sum()) if normalization == "shape" and mc.sum() > 0 else 1.0
        variance = model_data2 + mc2 * scale * scale
        supports[key] = variance > 0
        results[key] = (model_data, mc, scale, variance)
    common = supports["old"] & supports["new"]
    bins = int(np.count_nonzero(common))
    output = {"normalization": normalization, "valid_bins": bins,
              "valid_bin_indices": np.flatnonzero(common).tolist()}
    for key, (model_data, mc, scale, variance) in results.items():
        chi2 = float(np.sum((model_data[common] - mc[common] * scale) ** 2 / variance[common]))
        ndf = max(0, bins - (normalization == "shape"))
        output[key] = {"chi2": chi2, "ndf": ndf,
                       "chi2_ndf": chi2 / ndf if ndf else None,
                       "chi2_per_bin": chi2 / bins if bins else None,
                       "mc_scale": scale}
    output["delta_chi2_new_minus_old"] = output["new"]["chi2"] - output["old"]["chi2"]
    return output
