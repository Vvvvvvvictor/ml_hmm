"""Run the Gaussian jet-SF campaign and its fixed plot suite."""

import argparse

import os

from pathlib import Path

import sys


def scan(index):
    import awkward as ak
    import numpy as np
    import config
    from fit_model import (
        configure,
        signatures,
        compress,
        SignatureAccumulator,
        full_event_metadata,
    )
    from until import selection, event_weights
    from observables import reco_values
    from w_sf import mumet_pt, mumet_bins
    from root_io import input_batches, SCAN_STEP
    from time import perf_counter

    r = read(RUN / "inputs.json")["records"][index]
    era, region = r["era"], r["region"]
    data = r["sample"] == "data"
    m = read(RUN / era / "E/run_manifest.json")
    configure(era, m)
    branches = list(config.region_branches(region))
    if not data:
        branches += ["Jet_isGenMatched", *config.weight_branches(region)]
    accumulator = SignatureAccumulator()
    entries = selected = 0
    scale = None
    print("SCAN_ACCUMULATION balanced", flush=True)
    timings = {}
    started = perf_counter()
    for item in r["inputs"]:
        for a, scale in input_batches(
            Path(item["path"]),
            region,
            data,
            branches,
            compute_step=SCAN_STEP,
            statistics=timings,
        ):
            entries += len(a)
            a = a[selection(a, region)]
            selected += len(a)
            if not len(a):
                continue
            k = signatures(
                a, region, r["group"] == "DY", is_data=data, statistics=timings
            )
            ext = np.full((len(a), 117 if region == "wcr" else 107), 65535, np.uint16)
            ext[:, :105] = k
            ext[:, 105:107] = 0
            if region == "wcr":
                ext[:, 107:111] = 0
                ext[:, 112:116] = reco_values(a)
                ext[:, 116] = mumet_bins(mumet_pt(a))
            w = event_weights(a, scale, region, data)
            values = np.column_stack(
                [w, w * w, abs(w), np.ones(len(w)), w > 0, w < 0, abs(w)]
            ).astype(np.float64)
            accumulator.add(compress(ext, values))
            print("SCAN", index, entries, selected, flush=True)
    keys, values = accumulator.finish()
    if not selected:
        keys = np.empty((0, 117 if region == "wcr" else 107), dtype=np.uint16)
    print(
        "SCAN_TIMING",
        index,
        dict(
            timings,
            compute_s=perf_counter() - started - timings["open_s"] - timings["read_s"],
        ),
        flush=True,
    )
    dest = RUN / "shared/cache" / era / region / (r["sample"] + "__all.npz")
    save(dest, keys=keys, statistics=values)
    write(
        dest.with_suffix(".json"),
        dict(
            **r,
            partition="all",
            entries_read=entries,
            selected_entries=selected,
            retained_entries=selected,
            signatures=len(keys),
            nominal_scale=scale,
            model=m["model"],
            max_events=0,
            sumw=float(values[:, 0].sum()),
            config_sha256=m["configuration"]["config_sha256"],
            selection_signature=m["configuration"]["selection"]["selection_signature"],
            mumet_schema=2,
            scan_accumulation="balanced",
            jet_truncation={
                key: timings.get(key, 0)
                for key in ("truncated_events", "discarded_jets", "max_selected_jets")
            },
            scan_script_sha256=sha(Path(__file__)),
            **full_event_metadata(region, data),
        ),
    )


def preflight(era):
    import numpy as np
    from fit_model import configure, load_context, load_data, sf_prior_rows
    from conditional_model import model_and_jacobian

    m = read(RUN / era / "E/run_manifest.json")
    configure(era, m)
    x = np.zeros(794)
    slices = dict(
        jet=[0, 560],
        jet_matched=[0, 280],
        jet_unmatched=[280, 560],
        dy=[560, 690],
        w=[690, 794],
    )
    r, j = sf_prior_rows(x, m["fit_priors"], slices, np.ones(794, bool))
    assert r.shape == (560,) and not r.any() and j.shape == (560, 794)
    for region in m["model"]["fit_regions"]:
        mc = load_context(RUN / era / "E", m, era, region)
        data = load_data(RUN / era / "E", m, era, region)
        y, j = model_and_jacobian(x, mc)
        np.testing.assert_allclose(y, mc["h"].T @ mc["sumw"], rtol=1e-12, atol=1e-7)
        assert j.shape[1] == 794
        print("PREFLIGHT", era, region, len(y), flush=True)


def plot_setup():
    manifest = read(RUN / ERAS[0] / "E/run_manifest.json")
    write(
        RUN / "plotting_manifest.json",
        dict(
            records=read(RUN / "validation_inputs.json")["records"],
            fit_sha256={e: sha(RUN / e / "E/fit/fit_results.json") for e in ERAS},
            fit_regions=manifest["model"]["fit_regions"],
            validation_regions=manifest["model"]["validation_regions"],
            w_sf_mode="reco_njet_mumet_pt",
            match_split_plots=True,
        ),
    )


def extra(era):
    import numpy as np
    import matplotlib.pyplot as plt

    fit = read(RUN / era / "E/fit/fit_results.json")
    m = read(RUN / era / "E/run_manifest.json")
    p = fit["predictions"][era + "/wcr"]
    lo, hi = p["observation_blocks"]["mumet_pt_njet"]
    arrays = {
        k: np.asarray(p[k])[lo:hi].reshape(26, 4)
        for k in ("data", "nominal", "corrected")
    }
    errors = np.sqrt(
        np.diag(np.load(RUN / era / f"E/fit/{era}_wcr_observation_covariance.npy"))[
            lo:hi
        ]
    ).reshape(26, 4)
    out = RUN / era / "E/fit/diagnostics/wcr_mumet_pt_njet"
    out.mkdir(parents=True, exist_ok=True)
    for j, label in enumerate(m["model"]["w_njet_labels"]):
        fig, (ax, ratio) = plt.subplots(
            2,
            1,
            figsize=(11, 7),
            sharex=True,
            layout="constrained",
            gridspec_kw={"height_ratios": [3, 1]},
        )
        x = np.arange(26)
        ax.errorbar(x, arrays["data"][:, j], yerr=errors[:, j], fmt="k.", label="Data")
        for name, style in [("nominal", "--"), ("corrected", "-")]:
            y = arrays[name][:, j]
            ax.plot(x, y, style, label=name)
            pos = y > 0
            ratio.plot(x[pos], arrays["data"][pos, j] / y[pos], style, label=name)
        ratio.axhline(1, color="gray", linestyle="--")
        ax.set(title=f"{era} WCR Njet {label}", ylabel="Events")
        ax.legend()
        ratio.set(ylabel="Data / MC", xlabel="Muon+MET pT bin")
        for ext in ("png", "pdf"):
            fig.savefig(out / f"njet_{j}.{ext}", dpi=150)
        plt.close(fig)
    save(out / "plot_values.npz", **arrays, nominal_residual_sigma=errors)


def finish():
    import numpy as np
    from sf_export import verify_export

    summaries = {}
    lines = [
        "# Gaussian jet-SF fit",
        "",
        "Matched SF: 1 ± 0.30; unmatched SF: 1 ± 1.50, Gaussian in linear SF.",
        "Errors: local Gauss–Newton covariance including Gaussian priors; boundary flags do not denote calibrated confidence intervals.",
        "",
        "| Era | Reco χ² | Matched prior χ² | Unmatched prior χ² | Total | Rank | SF−error<0 cells |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for era in ERAS:
        verify_export(RUN / era / "E")
        fit = read(RUN / era / "E/fit/fit_results.json")
        assert (
            fit["optimizer_converged"] and fit["fit_success"] and fit["errors_valid"]
        ), era
        assert fit["sf_constraint_rows"] == 560
        np.testing.assert_allclose(
            fit["objective"], fit["reco_chi2"] + fit["sf_prior_chi2"], rtol=1e-12
        )
        summaries[era] = {
            k: fit[k]
            for k in [
                "reco_chi2",
                "sf_prior_chi2_by_type",
                "objective",
                "regularized_rank",
                "errors_valid",
                "boundary_parameter_indices",
                "covariance_method",
                "uncertainty_note",
            ]
        }
        prior = fit["sf_prior_chi2_by_type"]
        lines.append(
            f"| {era} | {fit['reco_chi2']:.3f} | {prior['jet_matched']:.3f} | {prior['jet_unmatched']:.3f} | {fit['objective']:.3f} | {fit['regularized_rank']} | {len(fit['boundary_parameter_indices'])} |"
        )
    missing = [
        f
        for f in read(RUN / "plots.json")["figures"]
        if not (RUN / f).is_file() or not (RUN / f).stat().st_size
    ]
    write(RUN / "figure_verification.json", dict(missing=missing, passed=not missing))
    assert not missing, missing
    from gallery import build

    build()
    (RUN / "REPORT.md").write_text("\n".join(lines) + "\n")
    write(
        RUN / "completion.json",
        dict(
            completed=True,
            fit_summary=summaries,
            figures=len(read(RUN / "plots.json")["figures"]),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument(
        "stage",
        choices=[
            "scan",
            "preflight",
            "fit",
            "export-sf",
            "metrics",
            "plot-setup",
            "histogram",
            "fit-histogram",
            "aggregate",
            "fit-plots",
            "log-plots",
            "extra",
            "ggf-histogram",
            "ggf-plots",
            "finish",
        ],
    )
    p.add_argument("--index", type=int)
    p.add_argument("--era")
    a = p.parse_args()
    os.environ["RECOIL_RUN_DIR"] = str(a.run_dir.resolve())
    os.environ["RECOIL_CODE_SNAPSHOT"] = str(Path(__file__).resolve().parents[2])
    from runtime import *

    if a.stage == "scan":
        scan(a.index)
    elif a.stage == "preflight":
        preflight(a.era)
    elif a.stage == "fit":
        from types import SimpleNamespace
        from fit_model import fit

        fit(
            SimpleNamespace(run_dir=RUN / a.era / "E", era=a.era),
            read(RUN / a.era / "E/run_manifest.json"),
        )
    elif a.stage == "export-sf":
        from sf_export import export_sf

        export_sf(RUN / a.era / "E")
    elif a.stage == "plot-setup":
        plot_setup()
    elif a.stage in ("metrics", "aggregate"):
        import plot_suite as v

        getattr(v, a.stage)(a.era)
    elif a.stage == "histogram":
        import validation_plots as v

        v.histogram(a.index)
    elif a.stage in ("fit-histogram", "fit-plots", "log-plots"):
        sys.path.insert(
            0, str(Path(os.environ["RECOIL_CODE_SNAPSHOT"]) / "fit_bin_plots")
        )
        import plot_eta_fit as v

        if a.stage == "fit-histogram":
            v.histogram(a.index)
        elif a.stage == "fit-plots":
            v.aggregate(a.era)
        else:
            import plot_eta_log as log

            log.main(a.era)
    elif a.stage == "extra":
        extra(a.era)
    elif a.stage == "ggf-histogram":
        from ggf_input_plots import histogram

        histogram((str(RUN), a.index, 0))
    elif a.stage == "ggf-plots":
        from ggf_input_plots import aggregate

        aggregate(a.era)
    else:
        finish()
