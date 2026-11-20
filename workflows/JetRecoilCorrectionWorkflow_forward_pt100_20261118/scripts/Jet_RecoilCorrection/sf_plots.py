from runtime import *

import config

import numpy as np


def draw_w_sf(fit, manifest, out, summary=None, fit_sha=None):
    import matplotlib.pyplot as plt

    values = np.asarray(fit["w_sf"]).reshape(26, 4)
    errors = np.asarray(fit["w_sf_error"], dtype=float).reshape(26, 4)
    valid = np.asarray(fit["w_sf_error_valid"]).reshape(26, 4)
    errors[~valid] = np.nan
    boundary = np.asarray(fit["w_sf_near_boundary"], dtype=bool).reshape(26, 4)
    labels = manifest["model"]["w_njet_labels"]
    edges = manifest["model"]["w_mumet_pt_edges"]
    pt_labels = [
        f"[{lo:g}, {hi:g})" if hi != "inf" else f"≥{lo:g}"
        for lo, hi in zip(edges[:-1], edges[1:])
    ]

    def store(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(out / (name + "." + ext), dpi=150)
        if summary is not None:
            summary.savefig(fig)
        plt.close(fig)

    out.mkdir(parents=True, exist_ok=True)
    from plots import _annotate_sf_values

    fig, ax = plt.subplots(figsize=(10, 15), layout="constrained")
    im = ax.imshow(values, aspect="auto", origin="lower", cmap="viridis")
    fig.colorbar(im, ax=ax, label="W SF")
    ax.set_xticks(range(4), labels)
    ax.set_yticks(range(26), pt_labels)
    ax.set(
        xlabel="Selected jet multiplicity",
        ylabel="Muon+MET pT bin [GeV]",
        title=f"{manifest['eras'][0]} {config.JET_SELECTION} W SF\nInformation rank {fit['rank']}/{fit['parameter_count']}; errors valid={fit['errors_valid']}; *: SF-error<0: {int(boundary.sum())} (local approximation)",
    )
    _annotate_sf_values(ax, im, values, np.arange(4), np.arange(26), errors=errors)
    store(fig, "candidate_w_sf")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained", sharex=True)
    for j, ax in enumerate(axes.ravel()):
        ax.plot(range(26), values[:, j], "o-")
        good = np.isfinite(errors[:, j])
        ax.errorbar(
            np.arange(26)[good],
            values[good, j],
            yerr=errors[good, j],
            fmt="none",
            capsize=2,
        )
        ax.axhline(1, color="gray", linestyle="--")
        ax.set(title="Njet " + labels[j], ylabel="W SF", xlabel="Muon+MET pT bin")
        ax.set_xticks(range(0, 26, 2), pt_labels[::2], rotation=70)
    store(fig, "candidate_w_sf_pt")
    write(
        out / "candidate_w_sf.json",
        dict(
            era=manifest["eras"][0],
            fit_sha256=fit_sha,
            w_sf=values.tolist(),
            w_sf_error=fit["w_sf_error"],
            w_sf_error_valid=fit["w_sf_error_valid"],
            w_sf_near_boundary=fit["w_sf_near_boundary"],
            uncertainty_note=fit["uncertainty_note"],
            pt_edges=edges,
            njet_edges=manifest["model"]["w_njet_edges"],
            njet_labels=labels,
            index="pt_bin * 4 + njet_bin",
            errors_valid=fit["errors_valid"],
            application_regions=["wcr"],
            catalog_group="W",
        ),
    )
