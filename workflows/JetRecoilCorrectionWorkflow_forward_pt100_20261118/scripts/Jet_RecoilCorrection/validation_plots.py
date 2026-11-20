"""Fill the standard postfit and regional validation histograms."""

from time import perf_counter

import sys

from pathlib import Path

import awkward as ak

import matplotlib

matplotlib.use("Agg")

import numpy as np

from scipy import sparse

import config

from runtime import RUN, read, save, sha as digest, write

sys.path.insert(0, str(Path(__file__).resolve().parent))

sys.path.insert(1, str(Path(__file__).resolve().parent.parent))

from fit_model import (
    configure,
    check_inputs,
    signatures,
    artifact_root,
    muon_observation_bins,
)

from until import selection, event_weights, dilepton_kinematics, bin_indices

from histograms import HISTOGRAMS, _jets, _jet_counts, _observable_counts

from histogram_accumulation import event_moments, one_hot_counts

from root_io import input_batches

from w_sf import (
    W_CLOSURE_EDGES,
    W_PLOT_EDGES,
    N_W_SF,
    mumet_pt,
    mumet_bins,
    w_sf_indices,
)

from conditional_model import observation_projection

import plots
import match_comparison


STAGES = {
    "wcr": ("nominal", "jet_sf", "w_sf", "jet_sf_w_sf"),
    "zee": ("nominal", "jet_sf", "dy_theory", "jet_sf_dy_theory"),
    "zcr": ("nominal", "jet_sf", "dy_theory", "jet_sf_dy_theory"),
    "srsb": ("nominal", "jet_sf", "dy_theory", "jet_sf_dy_theory"),
}


def run_dir(era):
    return RUN / era / "E"


def validation_dir(era):
    return run_dir(era) / "validation_candidate" / era


def histograms(region):
    import json

    spec = json.loads(
        (Path(__file__).resolve().parents[2] / "variables.json").read_text()
    )
    return {
        name: (np.asarray(v["edges"], dtype=float), v["label"])
        for name, v in spec["regions"][region].items()
    }


def plot_binning(region):
    return (
        {
            name: ["inf" if np.isposinf(v) else float(v) for v in edges]
            for name, edges in W_PLOT_EDGES.items()
        }
        if region == "wcr"
        else {}
    )


def fit_payload(era):
    m = read(run_dir(era) / "run_manifest.json")
    f = read(run_dir(era) / "fit/fit_results.json")
    configure(era, m)
    plots.ERA = era
    w_style(f)
    expected = read(RUN / "plotting_manifest.json")["fit_sha256"][era]
    if digest(run_dir(era) / "fit/fit_results.json") != expected:
        raise ValueError("Frozen fit changed during plotting")
    return m, f


def event_factors(jets, record, fit, ptll=None, wpt=None, mumet=None):
    njet = ak.to_numpy(ak.num(jets[4]))
    jet = np.ones(len(njet))
    theory = np.ones(len(njet))
    if record["sample"] != "data":
        sf = np.asarray(fit["jet_sf"]).ravel()
        if np.any(~np.isfinite(sf) | (sf <= 0)):
            raise ValueError("Invalid fitted per-jet SF")
        jet = np.exp(
            ak.to_numpy(
                ak.sum(
                    ak.unflatten(
                        np.log(sf[ak.to_numpy(ak.flatten(jets[4])).astype(int)])
                        * ak.to_numpy(ak.flatten((abs(jets[1]) > config.JET_SF_REGION["min_abs_eta_exclusive"]) & (jets[0] < config.JET_SF_REGION["max_pt_exclusive_gev"]))), njet
                    ),
                    axis=1,
                )
            )
        )
        if record["dy_sf_applicable"]:
            b, valid = bin_indices(ptll, config.DY_PT_EDGES)
            if not valid.all():
                raise ValueError("Invalid reconstructed dimuon pT")
            theory = np.asarray(fit["dy_sf"]).ravel()[
                b * config.DY_NJET_CATEGORIES + np.minimum(njet, 4)
            ]
        if record["group"] == "W":
            if record["region"] != "wcr":
                raise ValueError("W correction requires WCR")
            theory = np.asarray(fit["w_sf"])[w_sf_indices(njet, mumet_bins(mumet))]
    factors = np.column_stack([np.ones(len(jet)), jet, theory, jet * theory])
    if not np.isfinite(factors).all() or np.any(factors < 0) or np.any(theory <= 0):
        raise ValueError("Invalid event SF")
    return factors


def w_style(fit):
    label = "W reco Njet × muon+MET pT SF"
    plots.STAGE_STYLE["w_sf"] = (label, "#9467bd", ":")
    plots.STAGE_STYLE["jet_sf_w_sf"] = ("Jet SF × " + label, "#d45a32", "-")


def histogram(index):
    plotting_manifest = read(RUN / "plotting_manifest.json")
    r = plotting_manifest["records"][index]
    era, region = r["era"], r["region"]
    m, fit = fit_payload(era)
    check_inputs(r)
    data = r["sample"] == "data"
    axes = histograms(region)
    stages = STAGES[region]
    hist = {
        name + "__" + moment: np.zeros((4, len(edges) - 1))
        for name, (edges, _) in axes.items()
        for moment in ("sumw", "sumw2")
    }
    if region == "wcr":
        hist.update(
            {
                f"audit_{name}__{moment}": np.zeros((4, len(edges) - 1))
                for name, edges in W_CLOSURE_EDGES.items()
                for moment in ("sumw", "sumw2")
            }
        )
    hist.update(
        {
            f"cell__{moment}": np.zeros((4, config.N_PT * config.N_ETA))
            for moment in ("sumw", "sumw2")
        }
    )
    hist["matchcell"] = np.zeros((4, config.N_JET_CELLS))
    hist["event_yields"] = np.zeros((4, 2))
    hist["barrel_highpt"] = np.zeros((4, 2, 2))
    split = plotting_manifest["match_split_plots"]
    if split:
        for name in ("jet_pt", "jet_eta"):
            bins = len(axes[name][0]) - 1
            hist[name + "__split_sumw"] = np.zeros((4, 2 * bins))
            hist[name + "__split_covariance"] = np.zeros((4, 2 * bins, 2 * bins))
            hist[name + "__inclusive_covariance"] = np.zeros((4, bins, bins))
    if split:
        match_comparison.allocate(hist, axes)
    if region in m["model"]["fit_regions"]:
        size = (
            config.N_ZCR_OBS
            if region in config.Z_REGIONS
            else config.N_WCR_OBS
            + (4 + N_W_SF) * int(m["model"].get("w_sf_mode") == "reco_njet_mumet_pt")
            + muon_observation_bins(m["model"])
        )
        hist["fit_nominal"] = np.zeros(size)
        hist["fit_full"] = np.zeros(size)
    branches = list(config.region_branches(region))
    if not data:
        branches += ["Jet_isGenMatched", *config.weight_branches(region)]
    selected, read_entries, joined = 0, 0, 0
    timings = {}
    started = perf_counter()
    for item in r["inputs"]:
        path = Path(item["path"])
        for a, scale in input_batches(path, region, data, branches, statistics=timings):
            read_entries += len(a)
            a = a[selection(a, region)]
            if not len(a):
                continue
            selected += len(a)
            jets = _jets(a, data, apply_veto=False, statistics=timings)
            njet = ak.to_numpy(ak.num(jets[4]))
            ptll = dilepton_kinematics(a, region)[0] if region != "wcr" else None
            factors = event_factors(
                jets, r, fit, ptll, None, mumet_pt(a) if region == "wcr" else None
            )
            w = event_weights(a, scale, region, data)
            weights = w[:, None] * factors
            counts = _observable_counts(jets, ptll)
            if region == "zee":
                zpt, zphi, zmass = dilepton_kinematics(a, region)
                values = dict(
                    mass=zmass,
                    electron1_pt=np.asarray(a.ele1_pt),
                    electron2_pt=np.asarray(a.ele2_pt),
                    electron1_etaSC=np.asarray(a.ele1_etaSC),
                    electron2_etaSC=np.asarray(a.ele2_etaSC),
                )
                jx, jy = (
                    ak.to_numpy(ak.sum(jets[0] * f(jets[2]), axis=1))
                    for f in (np.cos, np.sin)
                )
                values["recoil_balance"] = np.hypot(
                    zpt * np.cos(zphi) + jx, zpt * np.sin(zphi) + jy
                ) / np.maximum(zpt, 1e-6)
                for name, value in values.items():
                    edges = axes[name][0]
                    indices = np.searchsorted(edges, value, side="right") - 1
                    valid = (
                        np.isfinite(value) & (indices >= 0) & (indices < len(edges) - 1)
                    )
                    counts[name] = one_hot_counts(indices, len(edges) - 1, valid)

            counts["cell"] = _jet_counts(
                jets[4] % (config.N_PT * config.N_ETA),
                np.arange(config.N_PT * config.N_ETA + 1) - 0.5,
            )
            closure_bins = None
            if region == "wcr":
                from observables import reco_values

                closure_bins = reco_values(a)
                display_bins = reco_values(a, edges=W_PLOT_EDGES)
                for j, name in enumerate(W_CLOSURE_EDGES):
                    counts[name] = one_hot_counts(
                        display_bins[:, j], len(axes[name][0]) - 1
                    )
                    counts["audit_" + name] = one_hot_counts(
                        closure_bins[:, j], len(W_CLOSURE_EDGES[name]) - 1
                    )
            for name, count in counts.items():
                sumw, sumw2, covariance = event_moments(
                    count, weights, covariance=split and name in ("jet_pt", "jet_eta")
                )
                hist[name + "__sumw"] += sumw
                hist[name + "__sumw2"] += sumw2
                if split and name in ("jet_pt", "jet_eta"):
                    hist[name + "__inclusive_covariance"] += covariance
            if split and not data:
                match_comparison.accumulate(hist, axes, jets, weights)
                matched = jets[4] < config.N_PT * config.N_ETA
                for name, jet_values in (("jet_pt", jets[0]), ("jet_eta", jets[1])):
                    c = sparse.hstack(
                        [
                            _jet_counts(jet_values[mask], axes[name][0])
                            for mask in (matched, ~matched)
                        ],
                        format="csr",
                    )
                    bins = c.shape[1] // 2
                    if (c[:, :bins] + c[:, bins:] != counts[name]).nnz:
                        raise AssertionError(
                            "Matched/unmatched counts differ from inclusive counts"
                        )
                    sumw, _, covariance = event_moments(c, weights, covariance=True)
                    hist[name + "__split_sumw"] += sumw
                    hist[name + "__split_covariance"] += covariance
            hist["event_yields"][:, 0] += weights.sum(axis=0)
            hist["event_yields"][:, 1] += (weights**2).sum(axis=0)
            if not data:
                cells = ak.to_numpy(ak.flatten(jets[4])).astype(int)
                for stage in range(4):
                    hist["matchcell"][stage] += np.bincount(
                        cells,
                        weights=np.repeat(weights[:, stage], njet),
                        minlength=config.N_JET_CELLS,
                    )
                pt, eta = jets[:2]
                unmatched = jets[4] >= config.N_PT * config.N_ETA
                local = np.column_stack(
                    [
                        ak.to_numpy(
                            ak.sum(
                                (abs(eta) < 1.5) & (pt >= lo) & (pt < hi) & label,
                                axis=1,
                            )
                        )
                        for lo, hi in [(70, 100), (100, np.inf)]
                        for label in (unmatched, ~unmatched)
                    ]
                )
                hist["barrel_highpt"] += (weights.T @ local).reshape(4, 2, 2)
            if region in m["model"]["fit_regions"]:
                key = signatures(a, region, r["group"] == "DY", is_data=data)
                extended = np.full((len(a), 117), 65535, np.uint16)
                extended[:, :105] = key
                if closure_bins is not None:
                    extended[:, 112:116] = closure_bins
                    extended[:, 116] = mumet_bins(mumet_pt(a))
                base = (
                    config.N_ZCR_OBS if region in config.Z_REGIONS else config.N_WCR_OBS
                )
                h = observation_projection(
                    extended,
                    base,
                    region,
                    m["model"].get("w_sf_mode") == "reco_njet_mumet_pt",
                    muon_observation_bins(m["model"]),
                )
                hist["fit_nominal"] += np.asarray(h.T @ w).ravel()
                hist["fit_full"] += np.asarray(h.T @ weights[:, -1]).ravel()
            print("HIST", index, read_entries, selected, flush=True)
    print(
        "HIST_TIMING",
        index,
        dict(
            timings,
            compute_s=perf_counter()
            - started
            - timings.get("open_s", 0)
            - timings.get("read_s", 0),
        ),
        flush=True,
    )
    if region in m["model"]["fit_regions"]:
        meta = read(
            artifact_root(run_dir(era), m, "cache")
            / "cache"
            / era
            / region
            / (r["sample"] + "__all.json")
        )
        assert selected == meta["selected_entries"]
        np.testing.assert_allclose(
            hist["event_yields"][0, 0], meta["sumw"], rtol=1e-9, atol=1e-6
        )
    out = validation_dir(era) / "raw" / region / (r["sample"] + ".npz")
    save(out, **hist)
    write(
        out.with_suffix(".json"),
        dict(
            record=r,
            selected=selected,
            entries_read=read_entries,
            joined=joined,
            fit_sha256=digest(run_dir(era) / "fit/fit_results.json"),
            stages=stages,
            jet_truncation={
                key: timings.get(key, 0)
                for key in ("truncated_events", "discarded_jets", "max_selected_jets")
            },
            match_split_plots=split,
            plot_binning=plot_binning(region),
            w_join=None,
        ),
    )
    print("HIST COMPLETE", index, selected, out, flush=True)
