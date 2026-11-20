"""Plot frozen ZCR/WCR jet observations, grouped by pT in each signed eta bin."""

import argparse

import sys

from pathlib import Path

import awkward as ak

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

import numpy as np

from scipy import sparse

from time import perf_counter

import os

RUN = Path(os.environ["RECOIL_RUN_DIR"])

sys.path.insert(
    0,
    str(Path(os.environ["RECOIL_CODE_SNAPSHOT"]) / "scripts/Jet_RecoilCorrection"),
)

import config
from runtime import read, save, sha as digest, write

from fit_model import configure, check_inputs

from until import selection, event_weights, dilepton_kinematics

from histograms import _jets

import validation_plots as plotting

from w_sf import mumet_pt

BLUE, YELLOW = "#377eb8", "#f2ce43"


def payload(era):
    manifest = read(RUN / era / "E/run_manifest.json")
    fit = read(RUN / era / "E/fit/fit_results.json")
    configure(era, manifest)
    fit_hash = digest(RUN / era / "E/fit/fit_results.json")
    assert fit_hash == read(RUN / "plotting_manifest.json")["fit_sha256"][era]
    return manifest, fit, fit_hash


def output_dir(era):
    return (
        RUN
        / era
        / "E/fit/diagnostics"
        / (
            "_".join(read(RUN / era / "E/run_manifest.json")["model"]["fit_regions"])
            + "_fit_bins"
        )
    )


def observations(arrays, jets, region):
    """Keep each jet's matching label associated with its recoil category."""
    cells = ak.to_numpy(ak.flatten(jets[4])).astype(int)
    njet = ak.to_numpy(ak.num(jets[4]))
    rows = np.repeat(np.arange(len(arrays)), njet)
    size = config.N_PT * config.N_ETA
    indices = cells % size
    if region in ("zee", "zcr", "wcr"):
        if region in config.Z_REGIONS:
            pt, phi, _ = dilepton_kinematics(arrays, region)
        else:
            pt = np.asarray(arrays["single_muon_pt"])
            phi = np.asarray(arrays["single_muon_phi"])
        delta = ak.to_numpy(ak.flatten(jets[2])) - np.repeat(phi, njet)
        recoil = (
            np.searchsorted(
                config.RECOIL_DPHI_EDGES[1:-1],
                np.abs(np.arctan2(np.sin(delta), np.cos(delta))),
                side="right",
            )
            + 1
        )
        recoil[np.repeat(pt, njet) < config.RECOIL_PT_THRESHOLDS[region]] = 0
        indices += recoil * size
    bins = size * 5
    split = sparse.csr_matrix(
        (np.ones(len(rows)), (rows, indices + (cells // size) * bins)),
        shape=(len(arrays), 2 * bins),
    )
    return split, split[:, :bins] + split[:, bins:]


def moments(split, inclusive, weights):
    bins = inclusive.shape[1]
    parts = np.asarray(split.T @ weights).T.reshape(2, 2, bins)
    variance = np.asarray(split.power(2).T @ weights**2).T.reshape(2, 2, bins)
    cross = np.asarray(split[:, :bins].multiply(split[:, bins:]).T @ weights**2).T
    total = np.asarray(inclusive.T @ weights).T
    total_variance = np.asarray(inclusive.power(2).T @ weights**2).T
    np.testing.assert_allclose(parts.sum(axis=1), total, rtol=1e-10, atol=1e-7)
    np.testing.assert_allclose(
        variance.sum(axis=1) + 2 * cross, total_variance, rtol=1e-10, atol=1e-7
    )
    return dict(
        parts=parts,
        variance=variance,
        cross=cross,
        total=total,
        total_variance=total_variance,
    )


def histogram(index):
    record = read(RUN / "plotting_manifest.json")["records"][index]
    era, region = record["era"], record["region"]
    manifest, fit, fit_hash = payload(era)
    assert region in manifest["model"]["fit_regions"]
    check_inputs(record)
    data = record["sample"] == "data"
    branches = list(config.region_branches(region))
    if not data:
        branches += ["Jet_isGenMatched", *config.weight_branches(region)]
    result = None
    entries = selected = 0
    timings = {}
    started = perf_counter()
    for item in record["inputs"]:
        for arrays, scale in plotting.input_batches(
            Path(item["path"]), region, data, branches, statistics=timings
        ):
            entries += len(arrays)
            arrays = arrays[selection(arrays, region)]
            selected += len(arrays)
            if not len(arrays):
                continue
            jets = _jets(arrays, data, apply_veto=False, statistics=timings)
            weight = event_weights(arrays, scale, region, data)
            ptll = (
                dilepton_kinematics(arrays, region)[0]
                if region in config.Z_REGIONS
                else None
            )
            correction = (
                np.ones(len(arrays))
                if data
                else plotting.event_factors(
                    jets,
                    record,
                    fit,
                    ptll,
                    mumet=mumet_pt(arrays) if region == "wcr" else None,
                )[:, 3]
            )
            weights = np.column_stack((weight, weight * correction))
            split, inclusive = observations(arrays, jets, region)
            values = moments(split, inclusive, weights)
            if result is None:
                result = {k: np.zeros_like(v) for k, v in values.items()}
            for key, value in values.items():
                result[key] += value
            print("HIST", index, entries, selected, flush=True)
    if result is None:
        bins = config.N_PT * config.N_ETA * 5
        result = moments(
            sparse.csr_matrix((0, 2 * bins)),
            sparse.csr_matrix((0, bins)),
            np.empty((0, 2)),
        )
    print(
        "FIT_HIST_TIMING",
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
    path = RUN / "fit_bin_plots/raw" / f"{index:03d}.npz"
    save(path, **result)
    write(path.with_suffix(".json"), dict(
        record=record,
        fit_sha256=fit_hash,
        script_sha256=digest(__file__),
        entries=entries,
        selected=selected,
        max_events=0,
    ))
    print("SAVED", path, flush=True)


def draw(era, eta, data, data_variance, parts, variance, cross, dest, pdf, wcr_scale):
    # All arrays have the ten-observation order within each pT group.
    bins = config.N_PT * 10
    edges, centers = np.arange(bins + 1) - 0.5, np.arange(bins)
    scale = np.tile([1] * 5 + [wcr_scale] * 5, config.N_PT)
    data, data_variance = data * scale, data_variance * scale**2
    parts, variance, cross = parts * scale, variance * scale**2, cross * scale**2
    total = parts.sum(axis=1)
    mc_variance = variance.sum(axis=1) + 2 * cross
    fig, (ax, ratio) = plt.subplots(
        2,
        1,
        figsize=(18, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3.3, 1]},
        layout="constrained",
    )
    positive, negative = np.zeros(bins), np.zeros(bins)
    for values, label, color in zip(
        parts[1], ("Postfit matched", "Postfit unmatched"), (BLUE, YELLOW)
    ):
        bottom = np.where(values >= 0, positive, negative)
        ax.bar(
            centers,
            values,
            bottom=bottom,
            width=1,
            color=color,
            label=label,
            linewidth=0,
        )
        positive += np.maximum(values, 0)
        negative += np.minimum(values, 0)
    positive, negative = np.zeros(bins), np.zeros(bins)
    for values, label, color in zip(
        parts[0], ("Prefit matched (stack)", "Prefit unmatched (stack)"), (BLUE, YELLOW)
    ):
        bottom = np.where(values >= 0, positive, negative)
        ax.stairs(
            bottom + values,
            edges,
            baseline=bottom,
            fill=False,
            color=color,
            linestyle="--",
            linewidth=1.2,
            label=label,
        )
        positive += np.maximum(values, 0)
        negative += np.minimum(values, 0)
    sigma = np.sqrt(np.maximum(mc_variance[1], 0))
    ax.fill_between(
        edges,
        np.r_[total[1] - sigma, (total[1] - sigma)[-1]],
        np.r_[total[1] + sigma, (total[1] + sigma)[-1]],
        step="post",
        color="gray",
        alpha=0.25,
        label="Postfit MC stat.",
    )
    ax.errorbar(
        centers,
        data,
        yerr=np.sqrt(data_variance),
        fmt="k.",
        markersize=3.5,
        label="Data",
        zorder=5,
    )
    valid = total[1] > 0
    ratio.errorbar(
        centers[valid],
        data[valid] / total[1, valid],
        yerr=np.sqrt(data_variance[valid]) / total[1, valid],
        fmt="k.",
        markersize=3,
    )
    relative = np.divide(sigma, total[1], out=np.full(bins, np.nan), where=valid)
    ratio.fill_between(
        edges,
        np.r_[1 - relative, (1 - relative)[-1]],
        np.r_[1 + relative, (1 + relative)[-1]],
        step="post",
        color="gray",
        alpha=0.25,
    )
    prefit = np.divide(total[0], total[1], out=np.full(bins, np.nan), where=valid)
    ratio.stairs(
        prefit,
        edges,
        color=BLUE,
        linestyle="--",
        linewidth=1.5,
        label="Prefit / postfit",
    )
    ratio.axhline(1, color="gray", linewidth=0.8)
    for pt in range(config.N_PT):
        lo, hi = config.JET_PT_EDGES[pt : pt + 2]
        label = f"{lo:g}–{hi:g} GeV" if np.isfinite(hi) else f"≥{lo:g} GeV"
        ax.text(
            (10 * pt + 5) / bins,
            1.025,
            label,
            transform=ax.transAxes,
            ha="center",
            fontsize=11,
        )
        for axis in (ax, ratio):
            axis.axvline(10 * pt + 4.5, color="gray", linestyle=":", linewidth=0.8)
            if pt:
                axis.axvline(10 * pt - 0.5, color="black", linewidth=1.3)
    ax.set_title(
        f"{era}   {config.ETA_EDGES[eta]:g} ≤ η < {config.ETA_EDGES[eta + 1]:g}", pad=37
    )
    ax.set_ylabel(f"Weighted jets / bin (WCR ×{wcr_scale:g})")
    ax.set_xlim(-0.5, bins - 0.5)
    ax.legend(ncol=3, fontsize=9, loc="best")
    ax.margins(y=0.22)
    ratio.set(ylabel="Data / postfit", ylim=(0, 2))
    ratio.legend(fontsize=9, loc="upper right")
    labels = [
        r"Z$_0$",
        r"Z$_1$",
        r"Z$_2$",
        r"Z$_3$",
        r"Z$_4$",
        *[rf"W$_{i}$" for i in range(5)],
    ] * config.N_PT
    ratio.set_xticks(centers, labels, fontsize=7)
    ratio.set_xlabel(
        "Z₀: pT(ll)<20 GeV; Z₁–Z₄: pT(ll)≥20 GeV, |Δφ(ll,jet)| in "
        "[0,1.5), [1.5,2.5), [2.5,2.9), [2.9,π]\n"
        "W₀: pT(μ)<40 GeV; W₁–W₄: pT(μ)≥40 GeV, same |Δφ(μ,jet)| intervals\n"
        "Fixed nominal regional normalization; statistical errors only"
    )
    for ext in ("png", "pdf"):
        fig.savefig(dest / f"eta_{eta:02d}.{ext}", dpi=150)
    pdf.savefig(fig)
    plt.close(fig)


def aggregate(era):
    wcr_scale = 0.1
    assert 0 < wcr_scale <= 1
    _, fit, fit_hash = payload(era)
    raw = RUN / "fit_bin_plots/raw"
    records = read(RUN / "plotting_manifest.json")["records"]
    combined = {}
    for region in fit["fit_regions"]:
        groups = {}
        for index, record in enumerate(records):
            if (record["era"], record["region"]) != (era, region):
                continue
            path = raw / f"{index:03d}.npz"
            meta = read(path.with_suffix(".json"))
            assert meta["record"] == record and meta["fit_sha256"] == fit_hash
            assert meta["script_sha256"] == digest(__file__)
            assert meta["max_events"] == 0
            kind = "data" if record["sample"] == "data" else "mc"
            with np.load(path) as source:
                if kind not in groups:
                    groups[kind] = {k: np.zeros_like(source[k]) for k in source.files}
                for key in source.files:
                    groups[kind][key] += source[key]
        alpha = fit["normalization_by_era"][era][region]
        for key in groups["mc"]:
            groups["mc"][key] *= alpha ** (
                2 if key in ("variance", "cross", "total_variance") else 1
            )
        bins = config.N_PT * config.N_ETA * 5
        data_event_reclassification = None
        if True:
            pred = fit["predictions"][era + "/" + region]
            data_delta = groups["data"]["total"][0] - pred["data"][:bins]
            mismatched_bins = int(np.count_nonzero(data_delta))
            data_event_reclassification = dict(
                mismatched_bins=mismatched_bins,
                maximum_absolute_difference=float(
                    np.max(np.abs(data_delta), initial=0.0)
                ),
                total_difference=float(data_delta.sum()),
            )
            np.testing.assert_array_equal(
                groups["data"]["total"][0], pred["data"][:bins]
            )
            for stage, name in enumerate(("nominal", "corrected")):
                np.testing.assert_allclose(
                    groups["mc"]["total"][stage],
                    pred[name][:bins],
                    rtol=2e-8,
                    atol=1e-5,
                )
        np.testing.assert_allclose(
            groups["mc"]["parts"].sum(axis=1),
            groups["mc"]["total"],
            rtol=1e-9,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            groups["mc"]["variance"].sum(axis=1) + 2 * groups["mc"]["cross"],
            groups["mc"]["total_variance"],
            rtol=1e-9,
            atol=1e-5,
        )
        combined[region] = groups
    dest = output_dir(era)
    dest.mkdir(parents=True, exist_ok=True)

    def join(kind, key):
        arrays = []
        for region in fit["fit_regions"]:
            values = combined[region][kind][key]
            n = 5
            arrays.append(
                values.reshape(
                    *values.shape[:-1], n, config.N_PT, config.N_ETA
                ).swapaxes(-3, -2)
            )
        return np.concatenate(arrays, axis=-2)

    values = dict(
        data=join("data", "total")[0],
        data_variance=join("data", "total_variance")[0],
        parts=join("mc", "parts"),
        variance=join("mc", "variance"),
        cross=join("mc", "cross"),
    )
    save(dest / "plot_values.npz", **values)
    with PdfPages(dest / "summary.pdf") as pdf:
        for eta in range(config.N_ETA):
            draw(
                era,
                eta,
                **{
                    k: v[..., eta].reshape(*v.shape[:-3], config.N_PT * 10)
                    for k, v in values.items()
                },
                dest=dest,
                pdf=pdf,
                wcr_scale=wcr_scale,
            )
    status = dict(
        era=era,
        eta_bins=config.N_ETA,
        pt_bins=config.N_PT,
        observations_per_pt=10,
        fit_sha256=fit_hash,
        script_sha256=digest(__file__),
        prediction_closure=True,
        mc_split_closure=True,
        data_event_reclassification=data_event_reclassification,
        normalization=fit["normalization_by_era"][era],
        axis_order="pt, [Z0,Z1,Z2,Z3,Z4,W0,W1,W2,W3,W4], signed_eta",
        figure_size_inches=[18, 7],
        stages=["prefit", "postfit"],
        components=["matched", "unmatched"],
        wcr_display_scale=wcr_scale,
        numerical_arrays_scaled=False,
        uncertainty="Event statistics including repeated jets and matched/unmatched covariance; no SF errors",
    )
    write(dest / "verification.json", status)
    print("AGGREGATE COMPLETE", era, dest, flush=True)
