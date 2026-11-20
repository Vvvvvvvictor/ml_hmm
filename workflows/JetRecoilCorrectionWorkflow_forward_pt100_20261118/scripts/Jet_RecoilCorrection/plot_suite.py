"""Joint ZCR/WCR fit metrics and regional validation with inclusive Data."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

import numpy as np

from runtime import RUN, REGIONS, read, save, write, sha

from fit_model import configure, load_context, load_data

from conditional_model import model_and_jacobian

from fit_statistics import finite, chi2

import config

import validation_plots as plotting

import plots
import match_comparison


def scale_hist(hist, alpha):
    out = {}
    for key, value in hist.items():
        power = 2 if key.endswith("__sumw2") or key.endswith("_covariance") else 1
        out[key] = value * alpha**power
        if key == "event_yields":
            out[key] = value * np.asarray([alpha, alpha**2])
    return out


def metrics(era):
    (RUN / "metrics").mkdir(parents=True, exist_ok=True)
    m = read(RUN / era / "E/run_manifest.json")
    fit = read(RUN / era / "E/fit/fit_results.json")
    configure(era, m)
    x = np.load(RUN / era / "E/fit/parameters.npy")
    assert len(x) == config.N_JET_CELLS + config.N_DY_CELLS + 104
    assert (
        fit["fit_regions"] == m["model"]["fit_regions"]
        and len(fit["w_sf"]) == 104
        and "w_pt_sf" not in fit
    )
    assert set(fit["predictions"]) == {era + "/" + r for r in fit["fit_regions"]}
    result = {}
    for region in fit["fit_regions"]:
        data = load_data(RUN / era / "E", m, era, region)
        mc0 = load_context(RUN / era / "E", m, era, region)
        alpha = float(data["sumw"].sum() / mc0["sumw"].sum())
        mc = load_context(RUN / era / "E", m, era, region, alpha)
        corrected, jac, details = model_and_jacobian(x, mc, return_details=True)
        if region in config.Z_REGIONS:
            np.testing.assert_array_equal(details["w"], 1.0)
        covariance = (
            (data["h"].T @ data["h"].multiply(data["sumw2"][:, None]))
            + (mc["h"].T @ mc["h"].multiply(mc["sums"][:, 1, None]))
        ).toarray()
        fit_covariance = np.load(
            RUN / era / "E/fit" / f"{era}_{region}_observation_covariance.npy"
        )
        covariance_matches_fit = bool(
            np.allclose(covariance, fit_covariance, rtol=2e-8, atol=1e-5)
        )
        covariance_difference = covariance - fit_covariance
        nominal = np.asarray(mc["h"].T @ mc["sumw"]).ravel()
        expected = dict(
            data=data["observed"].tolist(),
            nominal=nominal.tolist(),
            corrected=corrected.tolist(),
        )
        for key, value in expected.items():
            np.testing.assert_allclose(
                value, fit["predictions"][era + "/" + region][key], rtol=5e-7, atol=1e-5
            )
        stop = config.DY_OBS_OFFSET if region in config.Z_REGIONS else config.N_WCR_OBS
        out = dict(
            alpha=alpha,
            fit_region=True,
            predictions=expected,
            joint_chi2=chi2(data["observed"] - corrected, fit_covariance),
            jet_chi2=chi2(
                (data["observed"] - corrected)[:stop], fit_covariance[:stop, :stop]
            ),
            covariance_source="saved_with_fit",
            covariance_reconstruction_matches_fit=covariance_matches_fit,
            covariance_reconstruction_mismatched_elements=int(
                np.count_nonzero(
                    ~np.isclose(covariance, fit_covariance, rtol=2e-8, atol=1e-5)
                )
            ),
            covariance_reconstruction_max_absolute_difference=float(
                np.max(np.abs(covariance_difference), initial=0.0)
            ),
            maximum_event_sf=float(
                (details["raw"] * details["dy"] * details["w"]).max()
            ),
            w_sf=fit["w_sf"],
        )
        result[region] = out
        np.save(RUN / "metrics" / f"{era}_{region}_covariance.npy", fit_covariance)
        print("METRICS", era, region, flush=True)
    write(RUN / "metrics" / f"{era}.json", finite(result))


def match_split(era, region, data, total, out, summary, alpha):
    results = {}
    for name in ("jet_pt", "jet_eta"):
        edges, xlabel = plotting.histograms(region)[name]
        centers = (edges[:-1] + edges[1:]) / 2
        bins = len(centers)
        observed = data[name + "__sumw"][0]
        data_cov = data[name + "__inclusive_covariance"][0]
        np.testing.assert_allclose(
            np.diag(data_cov), data[name + "__sumw2"][0], rtol=1e-10, atol=1e-6
        )
        for norm in ("physical", "absolute", "shape"):
            for stage, index in (("nominal", 0), ("corrected", 3)):
                parts = total[name + "__split_sumw"][index].reshape(2, bins)
                cov = total[name + "__split_covariance"][index]
                projection = np.tile(np.eye(bins), (1, 2))
                full_cov = projection @ cov @ projection.T
                prediction = parts.sum(axis=0)
                np.testing.assert_allclose(
                    prediction, total[name + "__sumw"][index], rtol=1e-9, atol=1e-5
                )
                np.testing.assert_allclose(
                    full_cov,
                    total[name + "__inclusive_covariance"][index],
                    rtol=1e-9,
                    atol=1e-5,
                )
                np.testing.assert_allclose(
                    np.diag(full_cov),
                    total[name + "__sumw2"][index],
                    rtol=1e-9,
                    atol=1e-5,
                )
                if norm == "shape" and (prediction.sum() <= 0 or observed.sum() <= 0):
                    raise ValueError(
                        f"Invalid match-split normalization: {era}/{region}/{name}"
                    )
                factor = (
                    1 / alpha
                    if norm == "physical"
                    else float(observed.sum() / prediction.sum())
                    if norm == "shape"
                    else 1.0
                )
                parts = parts * factor
                cov = cov * factor**2
                full_cov = full_cov * factor**2
                prediction = prediction * factor
                fig, (ax, ratio_ax) = plt.subplots(
                    2,
                    1,
                    figsize=(10, 8),
                    sharex=True,
                    layout="constrained",
                    gridspec_kw={"height_ratios": [3, 1]},
                )
                positive = np.zeros(bins)
                negative = np.zeros(bins)
                for values, label, color in zip(
                    parts, ("Matched jets", "Unmatched jets"), ("#4c78a8", "#f2a541")
                ):
                    bottom = np.where(values >= 0, positive, negative)
                    ax.bar(
                        centers,
                        values,
                        width=np.diff(edges),
                        bottom=bottom,
                        label=label,
                        color=color,
                        align="center",
                    )
                    positive += np.maximum(values, 0)
                    negative += np.minimum(values, 0)
                sigma = np.sqrt(np.maximum(np.diag(full_cov), 0))
                ax.fill_between(
                    edges,
                    np.r_[prediction - sigma, (prediction - sigma)[-1]],
                    np.r_[prediction + sigma, (prediction + sigma)[-1]],
                    step="post",
                    color="gray",
                    alpha=0.3,
                    label="Total MC stat.",
                )
                ax.errorbar(
                    centers,
                    observed,
                    yerr=np.sqrt(np.maximum(np.diag(data_cov), 0)),
                    fmt="k.",
                    label="Inclusive Data",
                )
                valid = prediction > 0
                ratio_ax.errorbar(
                    centers[valid],
                    observed[valid] / prediction[valid],
                    yerr=np.sqrt(np.diag(data_cov)[valid]) / prediction[valid],
                    fmt="k.",
                )
                relative = np.divide(
                    sigma, prediction, out=np.full(bins, np.nan), where=valid
                )
                ratio_ax.fill_between(
                    edges,
                    np.r_[1 - relative, (1 - relative)[-1]],
                    np.r_[1 + relative, (1 + relative)[-1]],
                    step="post",
                    color="gray",
                    alpha=0.3,
                )
                ratio_ax.axhline(1, color="gray", linestyle="--")
                ratio_ax.set(xlabel=xlabel, ylabel="Data / MC", ylim=(0, 2))
                ax.set(
                    ylabel="Weighted jets / bin",
                    title=f"{era} {region.upper()} {stage}; {norm} normalization\nAll MC processes; per-jet matching; inclusive Data; statistical errors",
                )
                ax.legend()
                stem = (
                    out
                    / "match_split"
                    / (
                        f"{name}_{stage}"
                        if norm == "absolute"
                        else f"{name}_{stage}_{norm}"
                    )
                )
                stem.parent.mkdir(parents=True, exist_ok=True)
                for ext in ("png", "pdf"):
                    fig.savefig(stem.with_suffix("." + ext), dpi=150)
                summary.savefig(fig)
                plt.close(fig)
                results[name + "_" + stage + "_" + norm] = dict(
                    matched=parts[0],
                    unmatched=parts[1],
                    data=observed,
                    total_mc=prediction,
                    total_mc_variance=np.diag(full_cov),
                    matched_fraction=np.divide(
                        parts[0],
                        prediction,
                        out=np.full_like(prediction, np.nan),
                        where=prediction != 0,
                    ),
                    **chi2(observed - prediction, data_cov + full_cov),
                )
    write(out / "match_split/results.json", finite(results))


def render(era, region, manifest, data, groups, out, alpha):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    stages = plotting.STAGES[region]
    status = (
        "Fitted sample"
        if region in manifest["model"]["fit_regions"]
        else "Validation: no SF fit to this region"
    )
    fit_label = "/".join(manifest["model"]["fit_regions"]).upper()
    title = f"{era} {region.upper()} joint {fit_label} correction; {status}\nFixed nominal regional Data/MC normalization; statistical errors"
    result = {}
    total = {k: sum(g[k] for g in groups.values()) for k in data}
    with PdfPages(out / "agreement_summary.pdf") as pdf:
        for name, (edges, xlabel) in plotting.histograms(region).items():
            display = "mu_met_pt" if name == "recoil" else name
            if name == "recoil":
                xlabel = (
                    r"$p_T(\mu, p_T^{miss}) = |\vec p_T^\mu+\vec p_T^{miss}|$ [GeV]"
                )
            if name == "muon_pt":
                xlabel = r"Muon $p_T$ [GeV]"
            values = {
                g: {
                    stage: {
                        moment: v[name + "__" + moment][index]
                        for moment in ("sumw", "sumw2")
                    }
                    for index, stage in enumerate(stages)
                }
                for g, v in groups.items()
            }
            result[display] = {}
            for norm in ("physical", "absolute", "shape"):
                plot_groups = (
                    plots.scaled_groups(values, 1 / alpha)
                    if norm == "physical"
                    else values
                )
                norm_title = {
                    "physical": "Luminosity × cross-section normalization",
                    "absolute": "Fixed nominal regional Data/MC normalization",
                    "shape": "Each MC stage normalized to Data yield",
                }[norm]
                scales = plots.agreement(
                    name,
                    edges,
                    data[name + "__sumw"][0],
                    data[name + "__sumw2"][0],
                    plot_groups,
                    out / (display + "_" + norm),
                    title.replace(
                        "Fixed nominal regional Data/MC normalization", norm_title
                    )
                    + "; "
                    + norm,
                    stages,
                    norm,
                    xlabel,
                    pdf,
                )
                result[display][norm] = plots.agreement_metrics(
                    data[name + "__sumw"][0],
                    data[name + "__sumw2"][0],
                    plot_groups,
                    stages,
                    norm,
                )
                if any(s is None for s in scales.values()):
                    raise ValueError(
                        f"Nonpositive histogram area: {era}/{region}/{name}"
                    )
    with PdfPages(out / "match_split_summary.pdf") as pdf:
        match_split(era, region, data, total, out, pdf, alpha)
        match_comparison.render(era, region, total, out, pdf, alpha, plotting.histograms(region))
    save(
        out / "plot_values.npz",
        **{
            **{"data__" + k: v for k, v in data.items()},
            **{"mc_fixed_prefit__" + k: v for k, v in total.items()},
        },
        alpha=alpha,
    )
    write(out / "plot_metrics.json", finite(result))
    return total, result


def aggregate(era):
    plotting.RUN = RUN
    m, fit = plotting.fit_payload(era)
    records = [
        r for r in read(RUN / "plotting_manifest.json")["records"] if r["era"] == era
    ]
    cached = read(RUN / "metrics" / f"{era}.json")
    results = {}
    for region in REGIONS:
        selected = [r for r in records if r["region"] == region]
        raw = {}
        for r in selected:
            path = (
                plotting.validation_dir(era) / "raw" / region / (r["sample"] + ".npz")
            )
            meta = read(path.with_suffix(".json"))
            assert meta["record"] == r
            assert meta.get("plot_binning", {}) == plotting.plot_binning(region), (
                f"Stale validation binning: {path}"
            )
            assert meta["fit_sha256"] == sha(RUN / era / "E/fit/fit_results.json")
            with np.load(path) as f:
                raw[r["sample"]] = {k: f[k] for k in f.files}
        data = raw["data"]
        events = sum(
            raw[r["sample"]]["event_yields"][0, 0]
            for r in selected
            if r["sample"] != "data"
        )
        alpha = float(data["event_yields"][0, 0] / events)
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError(f"Invalid normalization {era}/{region}: {alpha}")
        groups = {}
        for r in selected:
            if r["sample"] == "data":
                continue
            values = scale_hist(raw[r["sample"]], alpha)
            group = groups.setdefault(r["group"], {})
            for key, value in values.items():
                group[key] = group.get(key, 0) + value
        out = plotting.validation_dir(era) / region
        total, plots_result = render(era, region, m, data, groups, out, alpha)
        if region in m["model"]["fit_regions"]:
            np.testing.assert_allclose(alpha, cached[region]["alpha"], rtol=1e-9)
            for key, value in [
                ("data", data["fit_nominal"]),
                ("nominal", total["fit_nominal"]),
                ("corrected", total["fit_full"]),
            ]:
                np.testing.assert_allclose(
                    value, cached[region]["predictions"][key], rtol=2e-8, atol=1e-5
                )
        pairs = total["barrel_highpt"].reshape(4, 2, 2)
        fractions = np.divide(
            pairs[:, :, 0],
            pairs.sum(axis=2),
            out=np.full((4, 2), np.nan),
            where=pairs.sum(axis=2) > 0,
        )
        results[region] = dict(
            alpha=alpha,
            fixed_all_stages=True,
            sf_derived_from_this_region=region in m["model"]["fit_regions"],
            uses_region_data_for_nominal_normalization=True,
            unmatched_barrel_highpt_fraction=fractions,
            plots=plots_result,
        )
        write(
            out / "normalization.json",
            finite({k: v for k, v in results[region].items() if k != "plots"}),
        )
    write(RUN / "validation_metrics" / f"{era}.json", finite(results))
    plots.ERA = era
    from sf_maps import sf_maps

    sf_maps(type("Args", (), {"run_dir": RUN / era / "E"})(), m)
    print("AGGREGATE COMPLETE", era, flush=True)
