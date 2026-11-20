#!/usr/bin/env python3
"""Compare baseline and GRL scores on the published nominal Run 3 outputs."""
import argparse
import json
from pathlib import Path

import numpy as np

from dnn_fit_campaign import write_json
from dnn_root_io import open_published_root

DY = "DY_105To160_ZpT-reweighted"
MASS = "diMufsr_kit_BSC_mass"
SCORE = "dnn_score_t"


def collect(root, years, categories):
    score = np.zeros((3, len(years), 100))
    sumw2 = np.zeros_like(score)
    # Inclusive, top 10% and top 1% of the signal-uniform score, for MC only.
    mass = np.zeros((3, len(years), 40))
    for category in categories:
        group = 0 if category == "data" else 1 if category == DY else 2
        with open_published_root(root / "outputs" / f"{category}.root") as source:
            for arrays in source["test"].iterate([MASS, SCORE, "eventWeight", "source_year"],
                                                 step_size="64 MB", library="np"):
                m, s, w, y = (arrays[name] for name in (MASS, SCORE, "eventWeight", "source_year"))
                if np.any(~np.isfinite(w)) or np.any(~np.isfinite(s)) or np.any((s < 0) | (s > 1)):
                    raise ValueError(f"Invalid scores or weights in {category}")
                sb = ((m >= 110) & (m <= 115)) | ((m >= 135) & (m <= 150))
                b = 100 * y[sb].astype(int) + np.minimum((100 * s[sb]).astype(int), 99)
                score[group] += np.bincount(b, weights=w[sb], minlength=100 * len(years)).reshape(-1, 100)
                sumw2[group] += np.bincount(b, weights=w[sb] ** 2, minlength=100 * len(years)).reshape(-1, 100)
                if group:
                    selected = (m >= 110) & (m <= 150)
                    b = 40 * y[selected].astype(int) + np.minimum((m[selected] - 110).astype(int), 39)
                    for i, threshold in enumerate((0.0, 0.9, 0.99)):
                        mask = s[selected] >= threshold
                        mass[i] += np.bincount(b[mask], weights=w[selected][mask],
                                               minlength=40 * len(years)).reshape(-1, 40)
        print(f"PT100 comparison {root.name}: {category}", flush=True)
    return score, sumw2, mass


def ratio_summary(score, sumw2):
    data, dy, other = score
    dv, dyv, ov = sumw2
    scale = data.sum() / (dy + other).sum()
    valid = dy > 0
    ratio = np.full(len(dy), np.nan)
    error = ratio.copy()
    ratio[valid] = (data[valid] - scale * other[valid]) / (scale * dy[valid])
    error[valid] = np.sqrt((dv[valid] + scale ** 2 * ov[valid]) / (scale * dy[valid]) ** 2
                           + ratio[valid] ** 2 * dyv[valid] / dy[valid] ** 2)
    data_shape = data / data.sum()
    mc_shape = (dy + other) / (dy + other).sum()
    return {
        "scale": float(scale),
        "shape_tv": float(np.abs(data_shape - mc_shape).sum() / 2),
        "ratio": [float(x) if np.isfinite(x) else None for x in ratio],
        "ratio_stat_error": [float(x) if np.isfinite(x) else None for x in error],
        "key_bins": {
            name: {"ratio": float(ratio[i]) if valid[i] else None,
                   "stat_error": float(error[i]) if valid[i] else None}
            for name, i in (("0.03–0.04", 3), ("0.99–1.00", 99))
        },
    }


def summarize(bundle, years):
    score, weights2, mass = bundle
    result = {}
    for index, year in enumerate([*years, "Run 3"]):
        s, w, m = ((score[:, index], weights2[:, index], mass[:, index]) if year != "Run 3"
                   else (score.sum(axis=1), weights2.sum(axis=1), mass.sum(axis=1)))
        normalizations = m.sum(axis=1)
        if np.any(normalizations <= 0):
            raise ValueError(f"Nonpositive mass normalization for {year}")
        result[year] = ratio_summary(s, w)
        result[year]["background_mass_unit_area"] = (m / normalizations[:, None]).tolist()
    return result


def fold_metrics(root, folds):
    result = {}
    for fold in folds:
        metrics = json.loads((root / "models_dnn" / f"DNN_metrics_ggH_{fold}.json").read_text())
        result[str(fold)] = {key: metrics[key] for key in (
            "best_epoch", "test_auc", "val_domain_bce", "val_domain_auc_abs_weight",
            "val_domain_bce_sample", "val_domain_flipped_bce_sample"
        ) if key in metrics}
    return result


def draw(path, summary, years):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(path) as pdf:
        for year in ("Run 3", *years):
            fig, (ratio_ax, mass_ax) = plt.subplots(1, 2, figsize=(12, 4.6))
            for label, color in (("baseline", "C0"), ("grl", "C1")):
                record = summary[label]["by_period"][year]
                ratio = np.array(record["ratio"], dtype=float)
                error = np.array(record["ratio_stat_error"], dtype=float)
                centers = (np.arange(100) + 0.5) / 100
                ratio_ax.errorbar(centers, ratio, yerr=error, fmt=".", ms=2, lw=0.75,
                                  color=color, label=label, alpha=0.8)
                shapes = record["background_mass_unit_area"]
                for i, (name, style) in enumerate((("inclusive", ":"), ("score ≥ 0.90", "--"),
                                                   ("score ≥ 0.99", "-"))):
                    mass_ax.stairs(shapes[i], np.arange(110, 151), color=color,
                                   linestyle=style, label=f"{label}, {name}")
            ratio_ax.axhline(1, color="0.45", linewidth=0.8)
            ratio_ax.set(xlabel="Signal-uniform score", ylabel="(Data − nonDY) / DY",
                         title="Sideband ratio with statistical error")
            mass_ax.set(xlabel=r"$m_{\mu\mu}$ (GeV)", ylabel="Unit-area background",
                        title="Background mass distribution")
            ratio_ax.legend()
            mass_ax.legend(fontsize=8)
            fig.suptitle(f"pt100 GRL comparison · {year}")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    years = manifest["years"]
    background = [name for name in manifest["categories"] if name not in manifest["signals"]]
    output = Path(manifest["artifacts"])
    result = {"score_edges": np.linspace(0, 1, 101).tolist(), "mass_edges": list(range(110, 151)),
              "mass_score_min": [0, 0.9, 0.99],
              "ratio_definition": "(Data - nonDY)/DY, MC normalized to the sideband Data yield in each period"}
    for label, root in (("baseline", Path(manifest["baseline"])), ("grl", output)):
        result[label] = {"test_fold_metrics": fold_metrics(root, manifest["defaults"]["folds"]),
                         "by_period": summarize(collect(root, years, background), years)}
    draw(output / "grl_comparison.pdf", result, years)
    write_json(output / "grl_comparison.json", result)


if __name__ == "__main__":
    main()
