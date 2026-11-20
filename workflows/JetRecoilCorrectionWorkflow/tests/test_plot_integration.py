"""Render real events with frozen SFs through all normalizations and match splits."""

import json, os, sys, tempfile
from pathlib import Path
import numpy as np
import awkward as ak
from scipy import sparse
from matplotlib.backends.backend_pdf import PdfPages

C = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C / "scripts/Jet_RecoilCorrection"))
run = Path(tempfile.mkdtemp(prefix="recoil_plot_check_"))
(run / "campaign.json").write_text(
    json.dumps(dict(eras=["2024"], validation_regions=["zee", "wcr", "zcr", "srsb"]))
)
os.environ["RECOIL_RUN_DIR"] = str(run)
import config

config.configure_reference("2024")
from samples import sample_records
from root_io import input_batches
from until import selection, event_weights, dilepton_kinematics
from histograms import _jets
import plots, plot_suite, ggf_input_plots as ggf
from validation_plots import event_factors
from histogram_accumulation import event_moments
from fit_statistics import finite

fit = json.loads(
    Path(
        "/eos/user/j/jiehan/HMuMuShare/JetRecoilCorrection_Run3/20261101T143656Z_zee_wcr/2024/E/fit/fit_results.json"
    ).read_text()
)
# The values come from the archived fit; this is an application/rendering regression only.
axes = ggf.AXES
split_data = {}
split_mc = {}
matched_totals = {}
checks = []
for data in (True, False):
    record = next(
        r
        for r in sample_records("zcr", "2024")
        if (r["sample"] == "data" if data else r["group"] == "DY")
    )
    branches = [*config.region_branches("zcr")]
    if not data:
        branches += ["Jet_isGenMatched", *config.weight_branches("zcr")]
    a, scale = next(
        input_batches(
            Path(record["inputs"][0]["path"]), "zcr", data, branches, entry_stop=3000
        )
    )
    a = a[selection(a, "zcr")]
    assert len(a) > 0
    jets = _jets(a, data, apply_veto=False)
    features, valid = ggf.features(a, jets, "2024")
    assert set(features) == set(axes)
    np.testing.assert_allclose(features["pTMu2oMmm"], features["pTMu1oMmm"])
    np.testing.assert_array_equal(features["source_year"], 2024)
    factors = event_factors(
        jets, dict(**record, region="zcr"), fit, dilepton_kinematics(a)[0]
    )
    weights = event_weights(a, scale, "zcr", data)[:, None] * factors
    for name in ("jet_pt", "jet_eta"):
        from histograms import _jet_counts, HISTOGRAMS

        values = jets[0 if name == "jet_pt" else 1]
        edges = HISTOGRAMS[name][0]
        bins = len(edges) - 1
        counts = _jet_counts(values, edges)
        part = [
            _jet_counts(values[jets[4] < 280], edges),
            _jet_counts(values[jets[4] >= 280], edges),
        ]
        split = sparse.hstack(part, format="csr")
        inclusive = counts
        h = np.asarray(inclusive.T @ weights).T
        h2 = np.asarray(inclusive.power(2).T @ (weights**2)).T
        parts = np.asarray(split.T @ weights).T
        cov = np.stack(
            [
                (split.T @ split.multiply((weights[:, i] ** 2)[:, None])).toarray()
                for i in range(4)
            ]
        )
        inc = np.stack(
            [
                (counts.T @ counts.multiply((weights[:, i] ** 2)[:, None])).toarray()
                for i in range(4)
            ]
        )
        dest = split_data if data else split_mc
        dest.update(
            {
                name + "__sumw": h,
                name + "__sumw2": h2,
                name + "__inclusive_covariance": inc,
            }
        )
        if not data:
            dest.update(
                {name + "__split_sumw": parts, name + "__split_covariance": cov}
            )
        if name == "jet_pt":
            if data:
                observed = h[0]
                observed2 = h2[0]
            else:
                groups = {
                    "DY": {
                        stage: dict(sumw=h[i], sumw2=h2[i])
                        for i, stage in enumerate(ggf.STAGES)
                    }
                }
    checks.append(("data" if data else "DY") + " real ROOT feature and SF application")
alpha = float(observed.sum() / sum(v["nominal"]["sumw"].sum() for v in groups.values()))
fixed = plots.scaled_groups(groups, alpha)
with PdfPages(run / "agreement.pdf") as pdf:
    for norm in ("physical", "absolute", "shape"):
        g = groups if norm == "physical" else fixed
        result = plots.agreement(
            "jet_pt",
            edges=np.linspace(25, 300, 56),
            data=observed,
            data2=observed2,
            groups=g,
            output=run / ("jet_pt_" + norm),
            title=norm,
            stages=ggf.STAGES,
            normalization=norm,
            xlabel="Jet pT [GeV]",
            summary=pdf,
        )
        assert all(x is not None for x in result.values())
        stats = plots.agreement_metrics(observed, observed2, g, ggf.STAGES, norm)
        assert set(stats) == set(ggf.STAGES)
# Match split covariance includes multiple jets per event and matched/unmatched cross terms.
fixed_mc = plot_suite.scale_hist(split_mc, alpha)
with PdfPages(run / "match_split_summary.pdf") as pdf:
    plot_suite.match_split("2024", "zcr", split_data, fixed_mc, run, pdf, alpha)
assert len(list((run / "match_split").glob("*.png"))) == 12
assert all(p.stat().st_size for p in run.rglob("*.pdf"))
checks.extend(
    [
        "Three normalization renderers and pull/ratio diagnostics",
        "Matched/unmatched sum, event covariance and 12 rendered comparisons",
    ]
)
print(json.dumps(dict(passed=True, checks=checks, output=str(run)), indent=2))

# Exercise the actual ggF worker and complete renderer with a bounded temporary run.
import hashlib
from validation_plots import histogram as regional_histogram
import validation_plots

model = json.loads((C / "model.json").read_text())
manifest = {
    **model,
    "eras": ["2024"],
    "configuration": {"selection": config.selection_metadata()},
    "jet_pt_mode": "nominal",
}
manifest_path = run / "2024/E/run_manifest.json"
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(manifest))
fit_path = run / "2024/E/fit/fit_results.json"
fit_path.parent.mkdir(parents=True)
fit_path.write_text(json.dumps(fit))
records = [
    dict(**r, era="2024", region="zcr")
    for r in sample_records("zcr", "2024")
    if r["sample"] == "data" or r["group"] == "DY"
]
(run / "plotting_manifest.json").write_text(
    json.dumps(
        dict(
            records=records,
            fit_sha256={"2024": hashlib.sha256(fit_path.read_bytes()).hexdigest()},
            match_split_plots=True,
        )
    )
)
original_batches = validation_plots.input_batches


# Restrict only this isolated test's reads; production has no event limit.
def bounded_batches(*args, **kwargs):
    yield from original_batches(*args, **kwargs, entry_stop=3000)


validation_plots.input_batches = bounded_batches
for index, r in enumerate(records):
    # For split data, bound the complete test population to the first input part.
    r["inputs"] = r["inputs"][:1]
(run / "plotting_manifest.json").write_text(
    json.dumps(
        dict(
            records=records,
            fit_sha256={"2024": hashlib.sha256(fit_path.read_bytes()).hexdigest()},
            match_split_plots=True,
        )
    )
)
for index in range(len(records)):
    regional_histogram(index)
    ggf.histogram((str(run), index, 3000))
validation_plots.input_batches = original_batches
raw = {}
for r in records:
    with np.load(
        run / "2024/E/validation_candidate/2024/raw/zcr" / (r["sample"] + ".npz")
    ) as handle:
        raw[r["sample"]] = dict(handle)
alpha = raw["data"]["event_yields"][0, 0] / sum(
    v["event_yields"][0, 0] for k, v in raw.items() if k != "data"
)
groups = {
    "DY": plot_suite.scale_hist(next(v for k, v in raw.items() if k != "data"), alpha)
}
plot_suite.render(
    "2024", "zcr", manifest, raw["data"], groups, run / "ordinary_render", alpha
)
ggf.render(run, "2024", "zcr", records, 3000)
assert (
    len(
        list(
            (run / "2024/E/validation_candidate/2024/ggf_dnn_inputs/zcr").glob("*.png")
        )
    )
    == 81
)
assert (run / "ordinary_render/plot_values.npz").is_file()
assert len(list((run / "ordinary_render/match_split").glob("*_jet_sf_physical.png"))) == 18
assert (run / "ordinary_render/match_split/jet_sf_comparisons.json").is_file()
print(
    json.dumps(
        dict(full_stage_render_passed=True, ggf_pngs=81, output=str(run)), indent=2
    )
)
