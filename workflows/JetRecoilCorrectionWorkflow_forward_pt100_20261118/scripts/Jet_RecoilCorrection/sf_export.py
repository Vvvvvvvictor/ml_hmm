"""Export fitted SFs as correctionlib schema-v2 files without rerunning the fit."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tempfile

import correctionlib
from correctionlib import schemav2
import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def edges(values, name):
    numbers = np.asarray(values, dtype=float)
    if (
        numbers.ndim != 1
        or len(numbers) < 2
        or np.any(np.isnan(numbers))
        or not np.all(np.diff(numbers) > 0)
        or not np.isfinite(numbers[0])
        or np.any(~np.isfinite(numbers[:-1]))
    ):
        raise ValueError(f"Invalid {name} edges")
    return ["inf" if np.isposinf(x) else float(x) for x in numbers]


def variable(name, kind, description):
    return dict(name=name, type=kind, description=description)


def grid(names, bins, values):
    return dict(
        nodetype="multibinning",
        inputs=names,
        edges=bins,
        content=values.ravel().tolist(),
        flow="error",
    )


def sf_array(fit, name, shape):
    """Validate the native fit shape before any flattening for correctionlib."""
    try:
        values = np.asarray(fit[name], dtype=float)
        uncertainty = np.asarray(fit[name + "_error"], dtype=float)
        valid = np.asarray(fit[name + "_error_valid"])
        boundary = np.asarray(fit[name + "_near_boundary"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid {name} values/errors") from exc
    if values.shape != shape or uncertainty.shape != shape or valid.shape != shape:
        raise ValueError(f"Unexpected {name} shape; expected {shape}")
    if np.any(~np.isfinite(values) | (values <= 0)):
        raise ValueError(f"Nonpositive or nonfinite {name}")
    if np.any(~np.isfinite(uncertainty) | (uncertainty < 0)):
        raise ValueError(f"Invalid {name} absolute errors")
    if valid.dtype.kind != "b" or not valid.all():
        raise ValueError(f"Invalid {name} error validity flags")
    if boundary.shape != shape or boundary.dtype.kind != "b" or not np.array_equal(boundary, values - uncertainty < 0):
        raise ValueError(f"Invalid {name} boundary flags")
    return values, uncertainty


def build_payload(manifest, fit):
    eras = manifest["eras"]
    if len(eras) != 1 or fit.get("eras") != eras:
        raise ValueError("Export requires matching single-era manifest and fit")
    for flag in ("optimizer_converged", "fit_success", "errors_valid"):
        if fit.get(flag) is not True:
            raise ValueError(f"Cannot export fit with {flag} != true")
    model = manifest["model"]
    if (
        model.get("dy_sf_mode") != "reco_ptll_njet"
        or model.get("dy_njet_categories") != 5
    ):
        raise ValueError("Unsupported SF parameterization")
    jet_pt = edges(model["jet_pt_edges"], "jet pt")
    jet_eta = edges(model["jet_eta_edges"], "jet eta")
    dy_pt = edges(model["dy_pt_edges"], "DY pt")
    jet_shape = (2, len(jet_pt) - 1, len(jet_eta) - 1)
    if np.asarray(fit["jet_sf_applicable"]).shape != jet_shape:
        raise ValueError("Unexpected jet_sf_applicable shape")
    fixed = np.asarray(fit["jet_sf_fixed"], dtype=bool)
    if fixed.shape != jet_shape:
        raise ValueError("Unexpected fixed jet SF shape")
    region = model["jet_sf_region"]
    eta_limit = float(region["min_abs_eta_exclusive"])
    pt_limit = float(region["max_pt_exclusive_gev"])
    pt_grid = np.asarray(jet_pt, dtype=float)
    eta_grid = np.asarray(jet_eta, dtype=float)
    if pt_limit not in pt_grid or eta_limit not in eta_grid or -eta_limit not in eta_grid:
        raise ValueError("Jet SF region boundaries must align with fit bins")
    eligible = (pt_grid[1:, None] <= pt_limit) & (np.minimum(abs(eta_grid[:-1]), abs(eta_grid[1:]))[None, :] >= eta_limit)
    expected_fixed = np.broadcast_to(~eligible, jet_shape)
    if not np.array_equal(fixed, expected_fixed):
        raise ValueError("Fixed jet SF cells disagree with the configured region")
    if np.any(np.asarray(fit["jet_sf"])[fixed] != 1.0) or np.any(np.asarray(fit["jet_sf_error"])[fixed] != 0.0):
        raise ValueError("Fixed jet SF cells must have value 1 and error 0")
    arrays = dict(
        jet=sf_array(fit, "jet_sf", jet_shape),
        dy=sf_array(fit, "dy_sf", (len(dy_pt) - 1, 5)),
    )
    # correctionlib binning is lower-inclusive. Add the single representable
    # double at +2.5 to the fixed region so the strict boundary is exact.
    eta_boundary = float(model["jet_sf_region"]["min_abs_eta_exclusive"])
    pt_boundary = float(model["jet_sf_region"]["max_pt_exclusive_gev"])
    if pt_boundary not in jet_pt or model["jet_sf_region"]["outside_sf"] != 1.0:
        raise ValueError("Invalid jet SF region for export")
    boundary_bin = jet_eta.index(eta_boundary)
    export_eta = jet_eta[:boundary_bin + 1] + [float(np.nextafter(eta_boundary, np.inf))] + jet_eta[boundary_bin + 1:]
    corrections = []
    for index, suffix in enumerate(("", "_error")):
        quantity = (
            "SF central value"
            if not suffix
            else "SF absolute one-sigma error; see original covariance for correlations"
        )
        common = dict(version=1, output=variable("value", "real", quantity))
        jet_values = arrays["jet"][index]
        boundary_value = 1.0 if index == 0 else 0.0
        jet_values = np.insert(jet_values, boundary_bin, boundary_value, axis=2)
        corrections.append(
            dict(
                **common,
                name="jet_sf" + suffix,
                description=f"{eras[0]} per-selected-jet {quantity}; signed eta; apply the original jet selection first",
                inputs=[
                    variable(
                        "is_gen_matched",
                        "int",
                        "Jet_isGenMatched: 1=matched, 0=unmatched",
                    ),
                    variable("pt", "real", "Corrected jet pT [GeV]"),
                    variable("eta", "real", "Signed corrected jet eta"),
                ],
                data=dict(
                    nodetype="category",
                    input="is_gen_matched",
                    content=[
                        dict(
                            key=matched,
                            value=grid(
                                ["pt", "eta"],
                                [jet_pt, export_eta],
                                jet_values[1 - matched],
                            ),
                        )
                        for matched in (0, 1)
                    ],
                ),
            )
        )
        corrections.append(
            dict(
                **common,
                name="dy_sf" + suffix,
                description=f"{eras[0]} {quantity}; direct DY only in {'/'.join(fit['dy_sf_applied_regions'])}; excludes DY to tau tau",
                inputs=[
                    variable("ptll", "real", "Reconstructed dilepton pT [GeV]"),
                    variable("njet", "int", "Number of selected reconstructed jets"),
                ],
                data=grid(
                    ["ptll", "njet"],
                    [dy_pt, [0.0, 1.0, 2.0, 3.0, 4.0, "inf"]],
                    arrays["dy"][index],
                ),
            )
        )
    payload = dict(
        schema_version=2,
        description=f"{eras[0]} fitted jet and DY SFs; original fit status: {fit.get('status', 'unspecified')}",
        corrections=corrections,
    )
    schemav2.CorrectionSet.model_validate(payload)
    check_runtime(
        correctionlib.CorrectionSet.from_string(json.dumps(payload, allow_nan=False)),
        payload,
    )
    return payload


def check_runtime(cset, payload):
    """Evaluate every exported cell through correctionlib's C++ evaluator."""
    for correction in payload["corrections"]:
        data = correction["data"]
        nodes = (
            [(entry["key"], entry["value"]) for entry in data["content"]]
            if data["nodetype"] == "category"
            else [(None, data)]
        )
        for category, node in nodes:
            points = []
            for name, bounds in zip(node["inputs"], node["edges"]):
                bounds = np.asarray(bounds, dtype=float)
                hi = np.where(np.isposinf(bounds[1:]), bounds[:-1] + 1.0, bounds[1:])
                midpoint = bounds[:-1] + (hi - bounds[:-1]) / 2.0
                points.append(
                    bounds[:-1].astype(np.int64) if name == "njet" else midpoint
                )
            arguments = [p.ravel() for p in np.meshgrid(*points, indexing="ij")]
            if category is not None:
                arguments.insert(0, category)
            np.testing.assert_allclose(
                cset[correction["name"]].evaluate(*arguments),
                node["content"],
                rtol=1e-14,
                atol=0,
            )


def sources(run_dir):
    run_dir = Path(run_dir)
    paths = dict(
        manifest=run_dir / "run_manifest.json",
        fit=run_dir / "fit/fit_results.json",
        exporter=Path(__file__).resolve(),
    )
    hashes = {name: sha256(path) for name, path in paths.items()}
    manifest, fit = read_json(paths["manifest"]), read_json(paths["fit"])
    payload = build_payload(manifest, fit)
    metadata = dict(
        era=manifest["eras"][0],
        schema_version=2,
        correction_version=1,
        sources={
            name: dict(path=str(path.resolve()), sha256=hashes[name])
            for name, path in paths.items()
        },
        selection=manifest["configuration"]["selection"],
        fit_status={
            key: fit[key]
            for key in ("optimizer_converged", "fit_success", "errors_valid")
        },
        application=dict(
            jet=dict(
                processes="all MC",
                regions=fit["validation_regions"],
                jet_selection=fit["jet_selection"],
            ),
            dy=dict(
                catalog_group="DY",
                excluded_processes=["DYTau"],
                regions=fit["dy_sf_applied_regions"],
            ),
        ),
        uncertainty=fit["uncertainty_note"],
        covariance_method=fit["covariance_method"],
        uncertainty_algorithm_sha256=fit["uncertainty_algorithm_sha256"],
        central_fit_tool_id=fit["central_fit_tool_id"],
        covariance_coordinates=fit["covariance_coordinates"],
        boundary_parameter_indices=fit["boundary_parameter_indices"],
        boundary_masks={name: fit[name] for name in ("jet_sf_near_boundary", "dy_sf_near_boundary", "w_sf_near_boundary") if name in fit},
        covariance_files={name: dict(path=str((paths["fit"].parent / name).resolve()), sha256=sha256(paths["fit"].parent / name)) for name in ("covariance.npy", "covariance_sf.npy")},
        correctionlib_version=correctionlib.__version__,
        verification=dict(
            cells_per_quantity=sum(
                len(c["data"]["content"])
                if c["data"]["nodetype"] != "category"
                else sum(len(x["value"]["content"]) for x in c["data"]["content"])
                for c in payload["corrections"]
                if not c["name"].endswith("_error")
            ),
            rtol=1e-14,
            atol=0,
        ),
    )
    if hashes != {name: sha256(path) for name, path in paths.items()}:
        raise ValueError("Export inputs changed during reading")
    return payload, metadata


def check_file(path, expected):
    actual = read_json(path)
    if actual != expected:
        raise ValueError(f"Export content mismatch: {path}")
    schemav2.CorrectionSet.model_validate(actual)
    check_runtime(correctionlib.CorrectionSet.from_file(str(path)), expected)


def verify_export(run_dir):
    payload, metadata = sources(run_dir)
    output = Path(run_dir) / "fit"
    files = ["corrections.json", "corrections.json.gz"]
    for name in files:
        check_file(output / name, payload)
    metadata["outputs"] = {name: sha256(output / name) for name in files}
    if read_json(output / "corrections_metadata.json") != metadata:
        raise ValueError("Correction export metadata or provenance mismatch")
    return metadata


def export_sf(run_dir):
    payload, metadata = sources(run_dir)
    output = Path(run_dir) / "fit"
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    # Publish only after both serialized representations load and evaluate.
    with tempfile.TemporaryDirectory(prefix=".sf_export_", dir=output) as tmp:
        tmp = Path(tmp)
        (tmp / "corrections.json").write_text(text)
        (tmp / "corrections.json.gz").write_bytes(gzip.compress(text.encode(), mtime=0))
        names = ["corrections.json", "corrections.json.gz"]
        for name in names:
            check_file(tmp / name, payload)
        metadata["outputs"] = {name: sha256(tmp / name) for name in names}
        (tmp / "corrections_metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False) + "\n"
        )
        for name in (*names, "corrections_metadata.json"):
            (tmp / name).replace(output / name)
    verify_export(run_dir)
    print(
        f"SF EXPORT {metadata['era']} {output} ({metadata['verification']['cells_per_quantity']} SFs and errors)",
        flush=True,
    )
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Single-era directory containing run_manifest.json and fit/",
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_export(args.run_dir)
        print(f"SF EXPORT VERIFIED {args.run_dir}", flush=True)
    else:
        export_sf(args.run_dir)
