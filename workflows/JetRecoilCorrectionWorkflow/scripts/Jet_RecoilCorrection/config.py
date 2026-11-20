"""Era inputs, reconstructed selections and fitted-SF binning."""

from pathlib import Path

import os

import hashlib

import json

BASE = Path("/eos/user/j/jiehan/HMuMuShare")

FIXED_REGION_GRADIENT_TOLERANCE = 1e-4

CORE = Path(__file__).resolve().parents[2]
ERAS = json.loads((CORE / "eras.json").read_text())

ERA = "2022EE"

JET_SELECTION = "inclusive"

SELECTION_VERSION = 8


INPUTS = {region: BASE / folder for region, folder in ERAS[ERA]["inputs"].items()}

LUMI_PB = ERAS[ERA]["lumi_pb"]


def configure(era, jet_selection="inclusive", *, jet_pt_edges=None):
    if era not in ERAS:
        raise ValueError(f"Unsupported data-taking era: {era}")
    if jet_selection != "inclusive":
        raise ValueError(f"Unsupported jet selection: {jet_selection}")
    global ERA, INPUTS, LUMI_PB, JET_SELECTION
    global \
        GEN_WEIGHT_BRANCH, \
        JET_WEIGHT_BRANCHES, \
        WCR_WEIGHT_BRANCHES, \
        SELECTION_VERSION
    global JET_PT_EDGES, N_PT, N_JET_CELLS, N_ZCR_OBS, N_WCR_OBS, N_OBS
    global DY_OBS_OFFSET, WCR_OBS_OFFSET
    ERA = era
    SELECTION_VERSION = 8
    JET_SELECTION = jet_selection
    INPUTS = {region: BASE / folder for region, folder in ERAS[era]["inputs"].items()}
    LUMI_PB = ERAS[era]["lumi_pb"]
    GEN_WEIGHT_BRANCH = "Generator_weight"
    JET_WEIGHT_BRANCHES = (
        GEN_WEIGHT_BRANCH,
        "puweight",
        "id_wgt_mu_1",
        "id_wgt_mu_2",
        "iso_wgt_mu_1",
        "iso_wgt_mu_2",
    )
    WCR_WEIGHT_BRANCHES = (
        GEN_WEIGHT_BRANCH,
        "puweight",
        "single_muon_id_wgt",
        "single_muon_iso_wgt",
    )
    JET_PT_EDGES = (25, 30, 35, 40, 50, 70, 100, float("inf"))
    if jet_pt_edges is not None:
        JET_PT_EDGES = tuple(float(x) for x in jet_pt_edges)
    N_PT = len(JET_PT_EDGES) - 1
    N_JET_CELLS = 2 * N_PT * N_ETA
    DY_OBS_OFFSET = N_RECOIL * N_PT * N_ETA
    N_ZCR_OBS = DY_OBS_OFFSET + N_DY_CELLS
    N_WCR_OBS = N_RECOIL * N_PT * N_ETA
    WCR_OBS_OFFSET = N_ZCR_OBS
    N_OBS = N_ZCR_OBS + N_WCR_OBS


TREE = "ntuple"

MAX_JETS = 32

SPLIT_SEED = 20261008

SPLIT_MODULUS = 5

FIT_FRACTION = 4

MC_VETO_CELLS = ()


def configure_reference(era, jet_selection="inclusive", *, jet_pt_edges=None):
    configure(era, jet_selection, jet_pt_edges=jet_pt_edges)


def analysis_metadata(region, is_data=False):
    payload = {
        "analysis_schema": 5,
        "max_selected_jets": MAX_JETS,
        "jet_overflow_policy": "stable_corrected_pt_descending_take_first_32",
        "jet_pt_mode": os.environ.get("CROWN_JET_PT_MODE", "nominal"),
        "jet_reader_schema": "compact_dual_jer_v2_jer_match",
        "jet_reader_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "Jet_DNN/jet_io.py").read_bytes()
        ).hexdigest(),
        "partition_rule": "all",
        "split_seed": SPLIT_SEED,
        "split_modulus": SPLIT_MODULUS,
        "fit_fraction": FIT_FRACTION,
        "weight_branches": [] if is_data else list(weight_branches(region)),
        "mc_veto_cells": [] if is_data else list(MC_VETO_CELLS),
        "zero_jet_events_retained": True,
        "strict_zero_sf_jet_removal": False,
    }
    return {
        **payload,
        "analysis_signature": hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest(),
    }


JET_PT_EDGES = (25, 30, 35, 40, 50, 70, 100, float("inf"))

ETA_POSITIVE_EDGES = (
    0,
    0.5,
    1,
    1.5,
    2,
    2.25,
    2.5,
    2.6,
    2.7,
    2.8,
    2.9,
    3,
    3.1,
    3.2,
    3.3,
    3.4,
    3.5,
    3.7,
    4,
    4.3,
    4.7,
)

ETA_EDGES = tuple(-x for x in ETA_POSITIVE_EDGES[:0:-1]) + ETA_POSITIVE_EDGES

RECOIL_DPHI_EDGES = (0.0, 1.5, 2.5, 2.9, 3.141592653589793)

DY_PT_EDGES = tuple(range(0, 101, 5)) + (110, 120, 140, 160, 200, float("inf"))

DY_NJET_CATEGORIES = 5

N_PT = len(JET_PT_EDGES) - 1

N_ETA = len(ETA_EDGES) - 1

N_RECOIL = 5

N_JET_CELLS = 2 * N_PT * N_ETA

N_DY_PT = len(DY_PT_EDGES) - 1

N_DY_CELLS = N_DY_PT * DY_NJET_CATEGORIES

N_ZCR_OBS = N_RECOIL * N_PT * N_ETA + N_DY_CELLS

N_WCR_OBS = N_RECOIL * N_PT * N_ETA

N_OBS = N_ZCR_OBS + N_WCR_OBS

DY_OBS_OFFSET = N_RECOIL * N_PT * N_ETA

WCR_OBS_OFFSET = DY_OBS_OFFSET + N_DY_CELLS

GEN_WEIGHT_BRANCH = "Generator_weight" if ERA == "2024" else "genWeight"

JET_WEIGHT_BRANCHES = (
    GEN_WEIGHT_BRANCH,
    "puweight",
    "id_wgt_mu_1",
    "id_wgt_mu_2",
    "iso_wgt_mu_1",
    "iso_wgt_mu_2",
)

WCR_WEIGHT_BRANCHES = (
    GEN_WEIGHT_BRANCH,
    "puweight",
    "single_muon_id_wgt",
    "single_muon_iso_wgt",
)

JET_BRANCHES = ("Jet_corrPt", "Jet_corrEta", "Jet_corrPhi", "Jet_corrMass")

ZCR_LEPTON_BRANCHES = tuple(
    f"{prefix}_{i}"
    for i in (1, 2)
    for prefix in (
        "pt_kit_bsc",
        "fsrPhoton_pt",
        "fsrPhoton_eta",
        "fsrPhoton_phi",
        "fsrPhoton_dROverEt2",
    )
) + tuple(f"mu{i}_fromH_{name}" for i in (1, 2) for name in ("eta", "phi"))

COMMON_BRANCHES = (
    *JET_BRANCHES,
    "run",
    "lumi",
    "event",
    "nbjets_loose",
    "nbjets_medium",
    "nmuons",
    "nelectrons",
)

ZCR_BRANCHES = (*COMMON_BRANCHES, *ZCR_LEPTON_BRANCHES, "trg_single_mu24", "H_mass")

WCR_BRANCHES = (*COMMON_BRANCHES, "single_muon_trigger_flag", "met_pt", "muon_met_mt")


def selection_metadata():
    payload = {
        "selection_version": SELECTION_VERSION,
        "max_selected_jets": MAX_JETS,
        "jet_overflow_policy": "stable_corrected_pt_descending_take_first_32",
        "event_selection_before_jet_truncation": True,
        "jet_selection": JET_SELECTION,
        "jet_selection_rule": {
            "inclusive": "corrected pT>25 GeV and |eta|<4.7",
            "jme": "corrected pT>25 GeV and |eta|<4.7; remove |eta|>2.5 and corrected pT<50 GeV",
            "he50": "corrected pT>25 GeV and |eta|<4.7; remove 2.5<|eta|<3.0 and corrected pT<50 GeV",
        }[JET_SELECTION],
        "bjet_veto": "nbjets_medium=0 and nbjets_loose<=1",
        "wcr_njet_selection": "Njet>=0, including zero selected-jet events",
        "jet_pt_edges": [
            "inf" if x == float("inf") else float(x) for x in JET_PT_EDGES
        ],
        "jet_eta_edges": list(map(float, ETA_EDGES)),
    }
    if ERAS[ERA]["adapter"] == "sampled":
        payload["btag_definition"] = {
            "discriminator": "Jet_btagUParTAK4B",
            "loose": 0.0499,
            "medium": 0.2605,
            "mask_selection": "CROWN good-jet, corrected-pT and |eta| selections",
        }
    payload["stored_btag_flags"] = True
    payload["input_adapter"] = ERAS[ERA]["adapter"]
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {**payload, "selection_signature": signature}


RECOIL_PT_THRESHOLDS = {"zcr": 20.0, "zee": 20.0, "wcr": 40.0}

RECOIL_AXES = {"zcr": "dilepton", "wcr": "muon"}

Z_REGIONS = ("zcr", "zee")
ZEE_BRANCHES = (
    *JET_BRANCHES,
    "run",
    "lumi",
    "event",
    "nbjets_loose",
    "nbjets_medium",
    "zee_pair_flag",
    "zee_os_flag",
    "zee_extra_veto_flag",
    "trg_single_ele30",
    "zee_mass",
    "zee_pt",
    "zee_phi",
    *[f"ele{i}_{v}" for i in (1, 2) for v in ("pt", "etaSC", "phi")],
)
ZEE_WEIGHT_BRANCHES = (
    "Generator_weight",
    "puweight",
    "id_wgt_ele_1",
    "id_wgt_ele_2",
    "reco_wgt_ele_1",
    "reco_wgt_ele_2",
    "trg_wgt_ele",
)


def region_branches(region):
    return {
        "zee": ZEE_BRANCHES,
        "zcr": ZCR_BRANCHES,
        "srsb": ZCR_BRANCHES,
        "wcr": (*WCR_BRANCHES, "single_muon_pt", "single_muon_phi", "met_phi"),
    }[region]


def weight_branches(region):
    return (
        ZEE_WEIGHT_BRANCHES
        if region == "zee"
        else (WCR_WEIGHT_BRANCHES if region == "wcr" else JET_WEIGHT_BRANCHES)
    )
