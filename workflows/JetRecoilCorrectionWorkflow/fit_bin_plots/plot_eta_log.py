"""Redraw saved fit observations with logarithmic main panels."""

import argparse

import time

from plot_eta_fit import *


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
    positive_values = np.r_[data[data > 0], total[total > 0]]
    lower = positive_values.min() / 3 if positive_values.size else 0.1
    upper = (
        max(
            np.max(data + np.sqrt(data_variance)),
            np.max(total[1] + np.sqrt(np.maximum(mc_variance[1], 0))),
            np.max(np.maximum(parts, 0).sum(axis=1)),
            1.0,
        )
        * 5
    )
    ax.set_yscale("log", nonpositive="clip")
    ax.set_ylim(lower, upper)
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
        np.maximum(np.r_[total[1] - sigma, (total[1] - sigma)[-1]], lower),
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
    if np.any(data <= 0):
        ax.scatter(
            centers[data <= 0],
            np.full(np.count_nonzero(data <= 0), lower * 1.12),
            marker="v",
            s=12,
            color="black",
            label="Data ≤0: axis floor",
            zorder=6,
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


def main(era):
    payload(era)
    source = output_dir(era)
    dest = source.parent / (source.name + "_log")
    dest.mkdir(parents=True, exist_ok=True)
    with np.load(source / "plot_values.npz") as z:
        values = {k: z[k] for k in z.files}
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
                wcr_scale=0.1,
            )
    (dest / "verification.json").write_text(
        json.dumps(dict(era=era, log_y=True, ratio_unchanged=True))
    )
