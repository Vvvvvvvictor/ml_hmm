"""Full-event reconstructed-SF application with the reference plotting helpers."""

import hashlib

import json

import matplotlib

matplotlib.use("Agg")

from matplotlib.backends.backend_pdf import PdfPages

import numpy as np

import config

from sample_groups import write_json

from fit_model import configure

STAGES = ("nominal", "jet_sf", "dy_theory", "jet_sf_dy_theory")

PV_EDGES = np.r_[np.arange(-0.5, 80.0, 1.0), np.inf]

N_NJET = 11


def finite_list(values):
    values = np.asarray(values, dtype=float)
    return np.where(np.isfinite(values), values, None).tolist()


def fit_result(run):
    path = run / "fit/fit_results.json"
    return json.loads(path.read_text()), hashlib.sha256(path.read_bytes()).hexdigest()


def sf_maps(args, manifest):
    era = manifest["eras"][0] if len(manifest["eras"]) == 1 else "combined"
    configure(era, manifest)
    from plots import _draw_dy_sf_map, _draw_sf_coverage
    from Jet_DNN.plotting_common import draw_sf_map

    result, fit_sha = fit_result(args.run_dir)
    output = args.run_dir / "fit/diagnostics/candidate_sf_maps"
    output.mkdir(parents=True, exist_ok=True)
    sf = np.asarray(result["jet_sf"])
    supported = np.ones(sf.size, bool)
    supported[
        np.asarray(result["unsupported_parameters"], int)[
            np.asarray(result["unsupported_parameters"], int) < sf.size
        ]
    ] = False
    supported = supported.reshape(sf.shape)
    applicable = np.asarray(
        result.get("jet_sf_applicable", np.ones(sf.shape, bool)), dtype=bool
    )
    supported &= applicable
    raw_errors = np.asarray(result["jet_sf_error"], dtype=float)
    error_valid = np.asarray(
        result.get("jet_sf_error_valid", np.isfinite(raw_errors)), dtype=bool
    )
    supported &= error_valid
    display_errors = np.where(supported, raw_errors, np.nan)
    _draw_sf_coverage(output, supported, applicable)
    with PdfPages(output / "summary.pdf") as summary:
        error_note = (
            "Errors include Gaussian constraints; valid="
            + str(result["regularized_errors_valid"])
            if result.get("fit_priors")
            else "Errors valid=" + str(result["errors_valid"])
        )
        boundary = np.asarray(result["jet_sf_near_boundary"], dtype=bool)
        error_note += f"; *: SF-error<0: {int(boundary.sum())} cells (local approximation)"
        dy_plot = {
            **result,
            "dy_plot_status": f"information rank {result['rank']}/{result['parameter_count']}; "
            f"errors valid={result['errors_valid']}",
        }
        _draw_dy_sf_map(output, dy_plot, summary, candidate=True)
        if "w_sf" in result:
            from sf_plots import draw_w_sf

            draw_w_sf(result, manifest, output, summary, fit_sha)
        for i, label in enumerate(("matched", "unmatched")):
            for applied in (False, True):
                values = np.where(
                    applicable[i] if applied else supported[i], sf[i], np.nan
                )
                stem = f"candidate_{'applied_' if applied else ''}jet_sf_{label}"
                draw_sf_map(
                    values,
                    config.ETA_EDGES,
                    config.JET_PT_EDGES,
                    output / stem,
                    f"{' + '.join(manifest['eras'])} {config.JET_SELECTION} {label}: UNPUBLISHED per-jet SF; Gaussian jet-SF priors\n{error_note}",
                    summary,
                    errors=display_errors[i],
                )
    write_json(
        output / "maps.json",
        {
            **result,
            "jet_sf_supported": supported.tolist(),
            "jet_sf_error": finite_list(display_errors),
            "uncertainty_note": result["uncertainty_note"],
            "jet_sf_applicable": applicable.tolist(),
            "coverage_note": "Support denotes nonzero sensitivity, not full identifiability",
            "dy_map_note": "Fitted shared DY SF indexed by dilepton pT and selected reco jet multiplicity",
            "display_note": "Colors follow the reference map range; numeric weights are not clipped",
        },
    )
    write_json(
        output / "candidate_dy_sf.json",
        dict(
            dy_sf=result["dy_sf"],
            dy_sf_error=result["dy_sf_error"],
            dy_sf_near_boundary=result["dy_sf_near_boundary"],
            uncertainty_note=result["uncertainty_note"],
            pt_edges=manifest["model"]["dy_pt_edges"],
            njet_categories=["0", "1", "2", "3", ">=4"],
            errors_valid=result["errors_valid"],
            regularized_errors_valid=result["regularized_errors_valid"],
            uncertainty_basis=result["covariance_method"],
            information_rank=result["rank"],
            parameter_count=result["parameter_count"],
            normalization_mode="free",
            application_regions=result["dy_sf_applied_regions"],
            definition="Shared positive DY SF indexed by reco dilepton pT and selected reco Njet",
        ),
    )
    print(f"SF MAPS {output}", flush=True)
