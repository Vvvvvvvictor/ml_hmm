"""Render recoil-correction products with the Jet DNN plotting helpers."""

import json

import sys

from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

SCRIPTS_DIR = Path(__file__).resolve().parents[1]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from Jet_DNN.plotting_common import GROUP_COLORS, draw_sf_map, draw_stacked_agreement

from config import (
    DY_NJET_CATEGORIES,
    DY_PT_EDGES,
    ERA,
    JET_SELECTION,
    ETA_EDGES,
    JET_PT_EDGES,
    selection_metadata,
)

COLORS = {
    **GROUP_COLORS,
    "W": "#9D755D",
    "MC": "#B279A2",
    "Signal": "#BAB0AC",
    "DYTau": "#B279A2",
}

STAGE_STYLE = {
    "nominal": ("Nominal MC", "#777777", "--"),
    "jet_sf": ("Jet SF", "#3569a8", "-"),
    "w_sf": ("W SF only", "#9467bd", ":"),
    "jet_sf_w_sf": ("Jet SF + W SF", "#d45a32", "-"),
    "dy_theory": ("DY only", "#9467bd", ":"),
    "jet_sf_dy_theory": ("Jet SF + DY", "#d45a32", "-"),
    "jet_sf_shape_only": ("Jet SF shape-only", "#1b9e77", "-."),
    "jet_sf_dy_theory_shape_only": ("Jet SF + DY shape-only", "#e6ab02", "-."),
}


def _annotate_sf_values(ax, image, values, x_centers, y_centers, errors=None):
    for (row, column), value in np.ndenumerate(values):
        if np.isfinite(value):
            red, green, blue, _ = image.cmap(image.norm(value))
            color = (
                "white"
                if 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.5
                else "black"
            )
            label = f"{value:.2g}" if 0 < abs(value) < 0.005 else f"{value:.2f}"
            if errors is not None:
                error = errors[row, column]
                label += f"\n± {error:.2g}" if np.isfinite(error) else "\nerror n/a"
                if np.isfinite(error) and value - error < 0:
                    label += " *"
            ax.text(
                x_centers[column],
                y_centers[row],
                label,
                ha="center",
                va="center",
                fontsize=7,
                color=color,
            )


def draw_signed_sf_map(values, eta_edges, pt_edges, output, title, summary):
    from matplotlib.colors import TwoSlopeNorm

    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    norm = TwoSlopeNorm(
        vcenter=0,
        vmin=min(-0.1, float(finite.min(initial=0))),
        vmax=max(0.1, float(finite.max(initial=0))),
    )
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#dddddd")
    fig, ax = plt.subplots(figsize=(25, 9), layout="constrained")
    im = ax.imshow(values, origin="upper", norm=norm, cmap=cmap, aspect="auto")
    ax.set_xticks(
        np.arange(len(eta_edges) - 1),
        [f"[{a:g}, {b:g})" for a, b in zip(eta_edges[:-1], eta_edges[1:])],
        rotation=90,
        fontsize=8,
    )
    ax.set_yticks(
        np.arange(len(pt_edges) - 1),
        [
            f"[{a:g}, {b:g})" if np.isfinite(b) else f"[{a:g}, ∞)"
            for a, b in zip(pt_edges[:-1], pt_edges[1:])
        ],
    )
    _annotate_sf_values(
        ax, im, values, np.arange(values.shape[1]), np.arange(values.shape[0])
    )
    ax.set(xlabel=r"Jet $\eta$ bin", ylabel=r"Jet $p_T$ bin [GeV]", title=title)
    fig.colorbar(im, ax=ax, label="Jet SF (signed)")
    for extension in ("pdf", "png"):
        fig.savefig(Path(output).with_suffix("." + extension), dpi=180)
    summary.savefig(fig)
    plt.close(fig)


def sf_maps(run_dir, *, candidate=False):
    run_dir = Path(run_dir)
    if candidate:
        fit = json.loads((run_dir / "fit/fit_candidate.json").read_text())
        output = run_dir / "fit/diagnostics/candidate_sf_maps"
        output.mkdir(parents=True, exist_ok=True)
        values = np.asarray(fit["jet_sf"])
        measured = np.ones(values.shape, dtype=bool)
        fixed = np.asarray(fit.get("unity_jet_parameters", []), dtype=int)
        measured.reshape(-1)[fixed] = False
        for index, reasons in fit.get("parameter_support_diagnostics", {}).items():
            if "no_support" in reasons:
                measured.reshape(-1)[int(index)] = False
        applicable = np.asarray(
            fit.get("jet_sf_applicable", np.ones(values.shape)), dtype=bool
        )
        _draw_sf_coverage(output, measured, applicable)
        draw_candidate = (
            draw_signed_sf_map
            if fit.get("jet_parameterization") == "direct_signed"
            else draw_sf_map
        )
        with PdfPages(output / "summary.pdf") as summary:
            _draw_dy_sf_map(output, fit, summary, candidate=True)
            for index, label in enumerate(("matched", "unmatched")):
                note = (
                    " (gray: unmeasured or inapplicable)"
                    if np.any(~measured[index])
                    else ""
                )
                draw_candidate(
                    np.where(
                        measured[index] & applicable[index], values[index], np.nan
                    ),
                    ETA_EDGES,
                    JET_PT_EDGES,
                    output / f"candidate_jet_sf_{label}",
                    f"{ERA} {JET_SELECTION} {label}: UNPUBLISHED candidate SF{note}",
                    summary,
                )
                draw_candidate(
                    np.where(applicable[index], values[index], np.nan),
                    ETA_EDGES,
                    JET_PT_EDGES,
                    output / f"candidate_applied_jet_sf_{label}",
                    f"{ERA} {JET_SELECTION} {label}: candidate SF values (inapplicable cells are not used)",
                    summary,
                )
        (output / "maps.json").write_text(
            json.dumps(
                {
                    **fit,
                    "jet_sf_measured": measured.tolist(),
                    "jet_sf_applicable": applicable.tolist(),
                },
                indent=2,
            )
            + "\n"
        )
        return
    fit = json.loads((run_dir / "fit/fit_results.json").read_text())
    output = run_dir / "sf_maps"
    output.mkdir(exist_ok=True)
    sf = np.asarray(fit["jet_sf"])
    errors = np.asarray(fit["jet_sf_error"])
    measured = np.asarray(fit["jet_sf_measured"], dtype=bool)
    applicable = np.asarray(fit.get("jet_sf_applicable", np.ones(sf.shape)), dtype=bool)
    _draw_sf_coverage(output, measured, applicable)
    with PdfPages(output / "sf_maps_summary.pdf") as summary:
        _draw_dy_sf_map(output, fit, summary, candidate=False)
        for index, label in enumerate(("matched", "unmatched")):
            note = (
                " (gray: unmeasured or inapplicable)"
                if np.any(~measured[index])
                else ""
            )
            draw_sf_map(
                np.where(measured[index] & applicable[index], sf[index], np.nan),
                ETA_EDGES,
                JET_PT_EDGES,
                output / f"jet_sf_{label}",
                f"{ERA} {JET_SELECTION} recoil correction: {label} jet SF{note}",
                summary,
            )
            draw_sf_map(
                sf[index],
                ETA_EDGES,
                JET_PT_EDGES,
                output / f"jet_sf_applied_{label}",
                f"{ERA} {JET_SELECTION} {label} SF values (inapplicable cells are not used)",
                summary,
            )
            display_pt = np.asarray(JET_PT_EDGES).copy()
            display_pt[-1] = display_pt[-2] + 40
            for kind, values in (
                ("absolute_error", errors[index]),
                ("relative_error", errors[index] / sf[index]),
                ("coverage", measured[index].astype(float)),
            ):
                fig, ax = plt.subplots(figsize=(16, 7), layout="constrained")
                cmap = plt.get_cmap("viridis").copy()
                cmap.set_bad("#dddddd")
                values = (
                    values
                    if kind == "coverage"
                    else np.where(measured[index] & applicable[index], values, np.nan)
                )
                image = ax.pcolormesh(
                    ETA_EDGES, display_pt, values, cmap=cmap, shading="flat"
                )
                ax.set(
                    xlabel="Signed jet eta",
                    ylabel="Jet pT [GeV]",
                    title=f"{ERA} {label} {kind}; fixed unity is unmeasured",
                )
                ax.set_yticks(
                    display_pt,
                    [str(v) if np.isfinite(v) else "inf" for v in JET_PT_EDGES],
                )
                fig.colorbar(image, ax=ax)
                for extension in ("png", "pdf"):
                    fig.savefig(output / f"{label}_{kind}.{extension}", dpi=150)
                summary.savefig(fig)
                plt.close(fig)
    (output / "maps.json").write_text(
        json.dumps(
            {
                "jet_sf": sf.tolist(),
                "measured": measured.tolist(),
                **{key: value for key, value in fit.items() if key.startswith("dy_sf")},
                "dy_normalization_mode": fit.get("dy_normalization_mode", "mean_one"),
                **selection_metadata(),
                "jet_sf_applicable": applicable.tolist(),
                "fixed_unity_parameters": fit.get("unity_jet_parameters", []),
                "jet_sf_error": fit["jet_sf_error"],
                "display_range": [0, 3],
                "display_note": "Color range only; numeric SF values are not clipped",
            },
            indent=2,
        )
        + "\n"
    )


def _draw_dy_sf_map(output, fit, summary, *, candidate):
    maps = fit["dy_sf_by_era"] if "dy_sf_by_era" in fit else {ERA: fit["dy_sf"]}
    for period, values in maps.items():
        _draw_period_dy_sf_map(
            output, fit, summary, period, values, candidate=candidate
        )


def _draw_period_dy_sf_map(output, fit, summary, period, values, *, candidate):
    values = np.asarray(values, dtype=float).reshape(-1, DY_NJET_CATEGORIES)
    errors = np.asarray(fit["dy_sf_error"], dtype=float).reshape(values.shape)
    boundary_count = int(np.asarray(fit["dy_sf_near_boundary"], dtype=bool).sum())
    fit = {**fit, "dy_plot_status": fit.get("dy_plot_status", "") + f"; SF-error<0: {boundary_count} cells (local approximation)"}
    fig, ax = plt.subplots(figsize=(10, 15), layout="constrained")
    image = ax.imshow(values, origin="lower", aspect="auto", cmap="viridis")
    _annotate_sf_values(
        ax,
        image,
        values,
        np.arange(DY_NJET_CATEGORIES),
        np.arange(len(values)),
        errors=errors,
    )
    ax.set_xticks(np.arange(DY_NJET_CATEGORIES), ("0", "1", "2", "3", "≥4"))
    ax.set_yticks(
        np.arange(len(values)),
        [
            f"[{a:g}, {b:g})" if np.isfinite(b) else f"≥{a:g}"
            for a, b in zip(DY_PT_EDGES[:-1], DY_PT_EDGES[1:])
        ],
    )
    note = (
        f"Errors include Gaussian constraints; *: SF-error<0 ({boundary_count} cells), local approximation"
        if fit.get("fit_priors")
        else fit.get("dy_plot_status", "")
    )
    ax.set(
        xlabel="Selected jet multiplicity",
        ylabel="Dilepton pT bin [GeV]",
        title=f"{period} {JET_SELECTION} DY SF\n{note}",
    )
    fig.colorbar(image, ax=ax, label="DY SF")
    stem = "candidate_dy_sf" if candidate else "dy_sf"
    if "dy_sf_by_era" in fit:
        stem += f"_{period}"
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{extension}", dpi=150)
    summary.savefig(fig)
    plt.close(fig)


def _draw_sf_coverage(output, measured, applicable):
    output.mkdir(parents=True, exist_ok=True)
    pt_edges = np.asarray(JET_PT_EDGES, dtype=float).copy()
    pt_edges[-1] = pt_edges[-2] + 40
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(("#777777", "#e6aa42", "#3d8b64"))
    with PdfPages(output / "coverage_summary.pdf") as summary:
        for index, label in enumerate(("matched", "unmatched")):
            values = np.where(~applicable[index], 0, np.where(measured[index], 2, 1))
            fig, ax = plt.subplots(figsize=(18, 7), layout="constrained")
            image = ax.pcolormesh(
                ETA_EDGES,
                pt_edges,
                values,
                cmap=cmap,
                vmin=-0.5,
                vmax=2.5,
                shading="flat",
            )
            ax.set_xticks(ETA_EDGES)
            ax.set(
                xlabel="Signed jet eta",
                ylabel="Jet pT [GeV]",
                title=f"{ERA} {label}: gray=inapplicable, amber=unity/unmeasured, green=fitted",
            )
            fig.colorbar(image, ax=ax, ticks=(0, 1, 2), label="Cell status")
            for extension in ("png", "pdf"):
                fig.savefig(output / f"coverage_{label}.{extension}", dpi=150)
            summary.savefig(fig)
            plt.close(fig)


def agreement_scales(data, predictions, normalization):
    if normalization not in ("physical", "absolute", "shape"):
        raise ValueError(f"Unknown agreement normalization: {normalization}")
    if normalization in ("physical", "absolute"):
        return {stage: 1.0 for stage in predictions}
    area = float(np.sum(data))
    scales = {}
    for stage, prediction in predictions.items():
        mc_area = float(np.sum(prediction))
        valid = np.isfinite(area) and area > 0 and np.isfinite(mc_area) and mc_area > 0
        scales[stage] = area / mc_area if valid else None
    return scales


def agreement(
    name,
    edges,
    data,
    data2,
    groups,
    output,
    title,
    stages,
    normalization,
    xlabel,
    summary,
    *,
    dy_applicable=True,
):
    edges = np.asarray(edges, dtype=float).copy()
    if np.isinf(edges[-1]):
        edges[-1] = edges[-2] + max(edges[-2] - edges[-3], 1)
    order = tuple(g for g in COLORS if g in groups)
    final = stages[-1]
    curves = []
    predictions = {
        stage: sum((groups[g][stage]["sumw"] for g in order), np.zeros_like(data))
        for stage in stages
    }
    scales = agreement_scales(data, predictions, normalization)
    if any(value is None for value in scales.values()):
        return scales
    for stage in stages:
        prediction = predictions[stage]
        variance = sum((groups[g][stage]["sumw2"] for g in order), np.zeros_like(data))
        scale = scales[stage]
        label, color, style = STAGE_STYLE[stage]
        if not dy_applicable and stage in ("dy_theory", "jet_sf_dy_theory"):
            label += " (DY not applicable)"
        denominator = data2 + variance * scale**2
        valid = denominator > 0
        chi2 = np.sum(
            (data[valid] - prediction[valid] * scale) ** 2 / denominator[valid]
        )
        ndof = max(0, int(valid.sum()) - (normalization == "shape"))
        label += f" ({chi2:.1f}/{ndof})"
        curves.append((label, prediction * scale, variance * scale**2, color, style))
    draw_stacked_agreement(
        edges,
        data,
        data2,
        {g: groups[g][final]["sumw"] for g in order},
        {g: groups[g][final]["sumw2"] for g in order},
        output,
        xlabel=xlabel,
        ylabel="Weighted jets / bin" if name.startswith("jet_") else "Events / bin",
        title=title,
        stages=curves,
        scale=scales[final],
        group_order=order,
        group_colors=COLORS,
        summary=summary,
        signed=True,
    )
    return scales


def scaled_groups(groups, factor):
    """Scale event yields and statistical variances with the same convention."""
    return {
        group: {
            stage: {
                moment: value * factor ** (2 if moment == "sumw2" else 1)
                for moment, value in values.items()
            }
            for stage, values in stages.items()
        }
        for group, stages in groups.items()
    }


def agreement_metrics(data, data2, groups, stages, normalization):
    predictions = {
        stage: sum((v[stage]["sumw"] for v in groups.values()), np.zeros_like(data))
        for stage in stages
    }
    scales = agreement_scales(data, predictions, normalization)
    result = {}
    for stage in stages:
        if scales[stage] is None:
            raise ValueError(f"Cannot normalize {stage}/{normalization}")
        prediction = predictions[stage] * scales[stage]
        variance = (
            data2
            + sum((v[stage]["sumw2"] for v in groups.values()), np.zeros_like(data2))
            * scales[stage] ** 2
        )
        valid = np.isfinite(variance) & (variance > 0)
        residual = data - prediction
        pulls = np.divide(
            residual,
            np.sqrt(np.maximum(variance, 0)),
            out=np.full_like(residual, np.nan),
            where=valid,
        )
        ratio = np.divide(
            data, prediction, out=np.full_like(data, np.nan), where=prediction > 0
        )
        result[stage] = dict(
            scale=scales[stage],
            chi2_diagonal=float(np.sum(pulls[valid] ** 2)),
            bins=int(valid.sum()),
            ndof=int(valid.sum()) - int(normalization == "shape"),
            pull_rms=float(np.sqrt(np.mean(pulls[valid] ** 2)))
            if valid.any()
            else None,
            data=data,
            mc=prediction,
            variance=variance,
            ratio=ratio,
            pull=pulls,
            note="Marginal statistical diagnostic; excludes SF uncertainty and other systematics",
        )
    return result
