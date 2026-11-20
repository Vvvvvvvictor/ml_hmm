"""Shared plotting styles and renderers for Jet DNN agreement products."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np

GROUP_ORDER = ("DY", "EWK", "Top", "Diboson", "Triboson")

GROUP_COLORS = {
    "DY": "#4C78A8",
    "EWK": "#F58518",
    "Top": "#54A24B",
    "Diboson": "#B279A2",
    "Triboson": "#E45756",
}

SCORE_COLORS = {"prefit": "#277da1", "sf_applied": "#f3722c", "jet_sf_zpt": "#54A24B"}

AGREEMENT_COLORS = ("#777777", "#3569a8", "#d45a32")


def unit_error_variance(values, variance, enabled=False):
    values = np.asarray(values, dtype=float)
    result = np.asarray(variance, dtype=float).copy()
    if enabled:
        result[(values == 0.0) & (result == 0.0)] = 1.0
    return result


def draw_sf_map(
    values,
    eta_edges,
    pt_edges,
    output,
    title,
    summary=None,
    *,
    errors=None,
    color_range=(0, 3),
    colorbar_label="Scale factor",
):
    values = np.asarray(values, dtype=float)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(25, 9))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#dddddd")
    image = ax.imshow(
        values,
        origin="upper",
        cmap=cmap,
        vmin=color_range[0],
        vmax=color_range[1],
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(eta_edges) - 1))
    ax.set_xticklabels(
        [f"[{a:g}, {b:g})" for a, b in zip(eta_edges[:-1], eta_edges[1:])],
        rotation=90,
        fontsize=8,
    )
    ax.set_yticks(np.arange(len(pt_edges) - 1))
    ax.set_yticklabels(
        [
            f"[{a:g}, {b:g})" if np.isfinite(b) else f"[{a:g}, ∞)"
            for a, b in zip(pt_edges[:-1], pt_edges[1:])
        ]
    )
    for (row, column), value in np.ndenumerate(values):
        if np.isfinite(value):
            label = f"{value:.2g}" if 0 < abs(value) < 0.005 else f"{value:.2f}"
            if errors is not None:
                error = errors[row, column]
                label += f"\n± {error:.2g}" if np.isfinite(error) else "\nσ n/a"
                if np.isfinite(error) and value - error < 0:
                    label += " *"
            red, green, blue, _ = image.cmap(image.norm(value))
            color = (
                "white"
                if 0.2126 * red + 0.7152 * green + 0.0722 * blue < 0.5
                else "black"
            )
            ax.text(
                column, row, label, ha="center", va="center", fontsize=7, color=color
            )
    ax.set(xlabel=r"Jet $\eta$ bin", ylabel=r"Jet $p_T$ bin [GeV]", title=title)
    fig.colorbar(image, ax=ax, label=colorbar_label, pad=0.015)
    fig.tight_layout()
    if summary is not None:
        summary.savefig(fig)
    for extension in ("pdf", "png"):
        fig.savefig(output.with_suffix(f".{extension}"), dpi=180)
    plt.close(fig)


def draw_stacked_agreement(
    edges,
    data,
    data2,
    group_sumw,
    group_sumw2,
    output,
    *,
    xlabel,
    ylabel,
    title,
    scale=1.0,
    stages=None,
    summary=None,
    xticks=None,
    xticklabels=None,
    group_order=GROUP_ORDER,
    group_colors=None,
    zero_bin_unit_error=False,
    signed=False,
):
    group_colors = GROUP_COLORS if group_colors is None else group_colors
    edges = np.asarray(edges, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    data = np.asarray(data, dtype=float)
    data2 = unit_error_variance(data, data2, zero_bin_unit_error)
    mc = sum(
        (np.asarray(group_sumw[group], dtype=float) for group in group_order),
        start=np.zeros_like(data),
    )
    mc2 = sum(
        (np.asarray(group_sumw2[group], dtype=float) for group in group_order),
        start=np.zeros_like(data),
    )
    mc2 = unit_error_variance(mc, mc2, zero_bin_unit_error)
    plotted_mc, plotted_mc2 = mc * scale, mc2 * scale**2
    figure, (axis, ratio_axis) = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        sharex=True,
        gridspec_kw={"height_ratios": (3.2, 1.0), "hspace": 0.05},
    )
    bottom = np.zeros_like(data)
    negative_bottom = np.zeros_like(data)
    ratios = []
    for group in group_order:
        values = np.asarray(group_sumw[group], dtype=float) * scale
        axis.bar(
            edges[:-1],
            values,
            width=widths,
            bottom=np.where(values < 0, negative_bottom, bottom) if signed else bottom,
            align="edge",
            color=group_colors[group],
            edgecolor="black",
            linewidth=0.25,
            alpha=0.82,
            label=group,
        )
        if signed:
            bottom += np.maximum(values, 0)
            negative_bottom += np.minimum(values, 0)
        else:
            bottom += values
    axis.fill_between(
        centers,
        plotted_mc - np.sqrt(np.maximum(plotted_mc2, 0)),
        plotted_mc + np.sqrt(np.maximum(plotted_mc2, 0)),
        step="mid",
        color="black",
        alpha=0.18,
        label="MC statistical uncertainty",
    )
    axis.errorbar(
        centers,
        data,
        yerr=np.sqrt(np.maximum(data2, 0)),
        fmt="ko",
        markersize=3,
        linewidth=0.7,
        label="Data",
        zorder=5,
    )
    if stages:
        for label, prediction, variance, color, linestyle in stages:
            prediction, variance = np.asarray(prediction), np.asarray(variance)
            axis.stairs(
                prediction, edges, color=color, linestyle=linestyle, label=label
            )
            axis.fill_between(
                centers,
                prediction - np.sqrt(np.maximum(variance, 0)),
                prediction + np.sqrt(np.maximum(variance, 0)),
                step="mid",
                color=color,
                alpha=0.10,
            )
            valid = prediction != 0 if signed else prediction > 0
            error = np.sqrt(
                data2[valid] / prediction[valid] ** 2
                + data[valid] ** 2 * variance[valid] / prediction[valid] ** 4
            )
            values = data[valid] / prediction[valid]
            ratio_axis.scatter(
                centers[valid][values > 2],
                np.full(np.count_nonzero(values > 2), 1.98),
                marker="^",
                s=18,
                color=color,
            )
            ratio_axis.scatter(
                centers[valid][values < 0],
                np.full(np.count_nonzero(values < 0), 0.02),
                marker="v",
                s=18,
                color=color,
            )
            ratios.extend(values.tolist())
            ratio_axis.errorbar(
                centers[valid],
                data[valid] / prediction[valid],
                yerr=error,
                fmt="o",
                markersize=2,
                color=color,
            )
    else:
        valid = plotted_mc != 0 if signed else plotted_mc > 0
        mc_band = np.zeros_like(plotted_mc)
        mc_band[valid] = np.sqrt(np.maximum(plotted_mc2[valid], 0)) / np.abs(
            plotted_mc[valid]
        )
        ratio = np.full_like(plotted_mc, np.nan)
        ratio_error = np.full_like(plotted_mc, np.nan)
        ratio[valid] = data[valid] / np.abs(plotted_mc[valid])
        ratio_error[valid] = np.sqrt(np.maximum(data2[valid], 0)) / np.abs(
            plotted_mc[valid]
        )
        ratio_axis.fill_between(
            centers, 1 - mc_band, 1 + mc_band, step="mid", color="gray", alpha=0.35
        )
        ratio_axis.errorbar(
            centers, ratio, yerr=ratio_error, fmt="o", color="black", markersize=3
        )
    axis.set(ylabel=ylabel, title=title)
    axis.legend(ncol=3, fontsize=8)
    ratio_axis.axhline(1, color="black", linewidth=0.8)
    ratio_axis.set(xlabel=xlabel, ylabel="Data / MC", ylim=(0.4, 1.6))
    ratio_axis.set_ylim(0, 2)
    # Out-of-range ratios are marked at the boundary; numerical products remain untouched.
    if xticks is not None:
        ratio_axis.set_xticks(xticks, xticklabels)
    ratio_axis.grid(axis="y", alpha=0.25)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output.with_suffix(".png"), dpi=150, bbox_inches="tight")
    if summary is not None:
        summary.savefig(figure, bbox_inches="tight")
    plt.close(figure)
