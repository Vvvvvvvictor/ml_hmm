"""Plot ggF DNN inputs in ZCR/SRSB with the frozen recoil-fit weights."""

import hashlib
import html
import json
import os
from pathlib import Path
import time

import awkward as ak
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

CORE = Path(__file__).resolve().parents[2]
SKIM = CORE / "definitions/ggf_skim.py"
MODEL = CORE / "definitions/ggf_model.json"
STAGES = ("nominal", "jet_sf", "dy_theory", "jet_sf_dy_theory")
PTDECOR = ("diMufsr_kit_BSC_pt", "jet1_pt", "jet2_pt", "dijet_pt")
REQUIRED_JETS = dict(
    jet1_pt=1,
    jet1_eta=1,
    jet2_pt=2,
    jet2_eta=2,
    dijet_mass=2,
    dijet_pt=2,
    dijet_eta=2,
    delta_eta_jj=2,
    delta_phi_jj=2,
    z_star=2,
    R_pT=2,
    min_delta_phi_dimu_jets=1,
    min_delta_eta_dimu_jets=1,
)


def lin(a, b, n):
    return np.linspace(a, b, n + 1)


AXES = {
    name: (np.asarray(v["edges"], dtype=float), v["label"])
    for name, v in json.loads((CORE / "variables.json").read_text())["ggf"].items()
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vector(pt, eta, phi, mass):
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    energy = np.sqrt(px * px + py * py + pz * pz + mass * mass)
    return np.column_stack((px, py, pz, energy))


def rapidity(v):
    return np.arctanh(v[:, 2] / v[:, 3])


def boost(v, beta):
    b2 = np.sum(beta * beta, axis=1)
    gamma = 1 / np.sqrt(1 - b2)
    dot = np.sum(beta * v[:, :3], axis=1)
    factor = (
        np.divide(gamma - 1, b2, out=np.zeros_like(b2), where=b2 > 0) * dot
        + gamma * v[:, 3]
    )
    return np.column_stack((v[:, :3] + factor[:, None] * beta, gamma * (v[:, 3] + dot)))


def collins_soper(mu1, mu2):
    """Reproduce calc_collins_soper_angles, including its Mu2/beam convention."""
    dimu = mu1 + mu2
    beta = -dimu[:, :3] / dimu[:, 3, None]
    beam = dimu[:, 3] / 2
    zero = np.zeros_like(beam)
    a = boost(np.column_stack((zero, zero, beam, beam)), beta)[:, :3]
    b = boost(np.column_stack((zero, zero, -beam, beam)), beta)[:, :3]

    def unit(v):
        return np.divide(
            v,
            np.linalg.norm(v, axis=1)[:, None],
            out=np.full_like(v, np.nan),
            where=np.linalg.norm(v, axis=1)[:, None] > 0,
        )

    z = unit(a - b)
    y = unit(np.cross(a, b))
    x = unit(np.cross(y, z))
    mu = boost(mu2, beta)[:, :3]
    phi = np.arctan2(np.sum(mu * y, axis=1), np.sum(mu * x, axis=1))
    phi = (phi + np.pi) % (2 * np.pi) - np.pi
    costheta = np.divide(
        np.sum(mu * z, axis=1),
        np.linalg.norm(mu, axis=1),
        out=np.full(len(mu), np.nan),
        where=np.linalg.norm(mu, axis=1) > 0,
    )
    return phi, costheta


def features(a, jets, era):
    mu = [
        vector(
            np.asarray(a[f"pt_kit_bsc_{i}"], float),
            np.asarray(a[f"mu{i}_fromH_eta"], float),
            np.asarray(a[f"mu{i}_fromH_phi"], float),
            0.1056584,
        )
        for i in (1, 2)
    ]
    bare = mu[0] + mu[1]
    fsr = bare.copy()
    for i in (1, 2):
        pt = np.asarray(a[f"fsrPhoton_pt_{i}"], float)
        good = (pt > 0) & (np.asarray(a[f"fsrPhoton_dROverEt2_{i}"]) < 0.012)
        fsr += vector(
            np.where(good, pt, 0),
            np.where(good, np.asarray(a[f"fsrPhoton_eta_{i}"], float), 0),
            np.where(good, np.asarray(a[f"fsrPhoton_phi_{i}"], float), 0),
            0.0,
        )
    ptll = np.hypot(fsr[:, 0], fsr[:, 1])
    mass = np.sqrt(np.maximum(fsr[:, 3] ** 2 - np.sum(fsr[:, :3] ** 2, axis=1), 0))
    etall = np.arcsinh(
        np.divide(fsr[:, 2], ptll, out=np.zeros_like(ptll), where=ptll > 0)
    )
    phill = np.arctan2(fsr[:, 1], fsr[:, 0])
    barept = np.hypot(bare[:, 0], bare[:, 1])
    bareeta = np.arcsinh(
        np.divide(bare[:, 2], barept, out=np.zeros_like(ptll), where=barept > 0)
    )
    barephi = np.arctan2(bare[:, 1], bare[:, 0])
    njet = ak.to_numpy(ak.num(jets[0]))
    order = ak.argsort(jets[0], axis=1, ascending=False)[:, :2]
    padded = [
        ak.to_numpy(ak.fill_none(ak.pad_none(v[order], 2, axis=1, clip=True), 0.0))
        for v in jets[:4]
    ]
    j1, j2 = [vector(*(v[:, i] for v in padded)) for i in (0, 1)]
    dijet = j1 + j2
    dijetpt = np.hypot(dijet[:, 0], dijet[:, 1])
    dijetmass = np.sqrt(
        np.maximum(dijet[:, 3] ** 2 - np.sum(dijet[:, :3] ** 2, axis=1), 0)
    )

    def dphi(x, y):
        return np.abs((x - y + np.pi) % (2 * np.pi) - np.pi)

    with np.errstate(divide="ignore", invalid="ignore"):
        zstar = (rapidity(bare) - (rapidity(j1) + rapidity(j2)) / 2) / np.abs(
            rapidity(j1) - rapidity(j2)
        )
        balance = np.hypot(bare[:, 0] + dijet[:, 0], bare[:, 1] + dijet[:, 1]) / (
            barept + padded[0].sum(axis=1)
        )
    phi, costheta = collins_soper(*mu)
    result = dict(
        diMufsr_kit_BSC_pt=ptll,
        diMufsr_kit_BSC_eta=etall,
        mu1_fromH_eta=np.asarray(a.mu1_fromH_eta),
        mu2_fromH_eta=np.asarray(a.mu2_fromH_eta),
        phi_CS=phi,
        cos_theta_CS=costheta,
        pTMu1oMmm=np.asarray(a.pt_kit_bsc_1) / mass,
        # Preserve the existing training-ntuple definition; do not silently fix it.
        pTMu2oMmm=np.asarray(a.pt_kit_bsc_1) / mass,
        njets=njet,
        z_star=zstar,
        R_pT=balance,
        dijet_mass=dijetmass,
        dijet_pt=dijetpt,
        dijet_eta=np.arcsinh(
            np.divide(dijet[:, 2], dijetpt, out=np.zeros_like(ptll), where=dijetpt > 0)
        ),
        delta_eta_jj=np.abs(padded[1][:, 0] - padded[1][:, 1]),
        delta_phi_jj=dphi(padded[2][:, 0], padded[2][:, 1]),
        jet1_pt=padded[0][:, 0],
        jet1_eta=padded[1][:, 0],
        jet2_pt=padded[0][:, 1],
        jet2_eta=padded[1][:, 1],
        min_delta_phi_dimu_jets=np.where(
            njet == 1,
            dphi(phill, padded[2][:, 0]),
            np.minimum(dphi(barephi, padded[2][:, 0]), dphi(barephi, padded[2][:, 1])),
        ),
        min_delta_eta_dimu_jets=np.where(
            njet == 1,
            np.abs(etall - padded[1][:, 0]),
            np.minimum(
                np.abs(bareeta - padded[1][:, 0]), np.abs(bareeta - padded[1][:, 1])
            ),
        ),
        source_year=np.full(len(njet), int(era[:4])),
    )
    valid = {
        name: np.isfinite(value) & (njet >= REQUIRED_JETS.get(name, 0))
        for name, value in result.items()
    }
    for name in PTDECOR:
        result[name + "__over_mass"] = result[name] / mass
        valid[name + "__over_mass"] = valid[name] & np.isfinite(mass) & (mass > 0)
    return result, valid


def output_dir(run, era):
    return Path(run) / era / "E/validation_candidate" / era / "ggf_dnn_inputs"


from root_io import input_batches


def histogram(task):
    run, index, limit = task
    os.environ["RECOIL_RUN_DIR"] = str(run)
    import config
    from runtime import save
    from fit_model import check_inputs
    from validation_plots import fit_payload, event_factors
    from histograms import _jets
    from until import selection, event_weights, dilepton_kinematics

    transport = "xrootd"
    records = read(Path(run) / "plotting_manifest.json")["records"]
    record = records[index]
    era, region = record["era"], record["region"]
    manifest, fit = fit_payload(era)
    check_inputs(record)
    data = record["sample"] == "data"
    output = output_dir(run, era) / "raw" / region / (record["sample"] + ".npz")
    provenance = dict(
        record=record,
        fit_sha256=sha(Path(run) / era / "E/fit/fit_results.json"),
        code_sha256=sha(__file__),
        max_entries=limit,
        transport=transport,
    )
    metadata = output.with_suffix(".json")
    if metadata.is_file() and output.is_file():
        saved = read(metadata)
        if saved.get("provenance") == provenance and saved["artifact_sha256"] == sha(output):
            return dict(region=region, sample=record["sample"], cached=True)
    branches = list(config.ZCR_BRANCHES)
    if not data:
        branches += ["Jet_isGenMatched", *config.JET_WEIGHT_BRANCHES]
    hist = {
        name + "__" + moment: np.zeros((4, len(edges) - 1))
        for name, (edges, _) in AXES.items()
        for moment in ("sumw", "sumw2")
    }
    hist["event_yields"] = np.zeros((4, 2))
    hist["population"] = np.zeros(
        (len(AXES), 3, 4)
    )  # valid / underflow / overflow, per stage
    hist["missing"] = np.zeros((len(AXES), 4))
    hist["ptll_reference"] = np.zeros((4, len(config.DY_PT_EDGES) - 1))
    selected = entries = 0
    jet_statistics = {}
    started = time.monotonic()
    for item in record["inputs"]:
        stop = max(0, limit - entries) if limit else None
        if limit and not stop:
            break
        for a, scale in input_batches(
            Path(item["path"]), region, data, branches, entry_stop=stop
        ):
            entries += len(a)
            a = a[selection(a, region)]
            if not len(a):
                continue
            selected += len(a)
            jets = _jets(a, data, apply_veto=False, statistics=jet_statistics)
            values, valid = features(a, jets, era)
            ptll = dilepton_kinematics(a)[0]
            factors = event_factors(jets, record, fit, ptll)
            weights = event_weights(a, scale, region, data)[:, None] * factors
            hist["event_yields"][:, 0] += weights.sum(axis=0)
            hist["event_yields"][:, 1] += (weights * weights).sum(axis=0)
            for j, (name, (edges, _)) in enumerate(AXES.items()):
                value, keep = values[name], valid[name]
                bins = np.clip(
                    np.searchsorted(edges, value[keep], side="right") - 1,
                    0,
                    len(edges) - 2,
                )
                # Fold finite tails into endpoint bins, retaining every valid event.
                for s in range(4):
                    w = weights[keep, s]
                    hist[name + "__sumw"][s] += np.bincount(
                        bins, weights=w, minlength=len(edges) - 1
                    )
                    hist[name + "__sumw2"][s] += np.bincount(
                        bins, weights=w * w, minlength=len(edges) - 1
                    )
                hist["population"][j, 0] += weights[keep].sum(axis=0)
                hist["population"][j, 1] += weights[keep & (value < edges[0])].sum(
                    axis=0
                )
                hist["population"][j, 2] += weights[keep & (value >= edges[-1])].sum(
                    axis=0
                )
                hist["missing"][j] += weights[~keep].sum(axis=0)
            b = np.searchsorted(config.DY_PT_EDGES, ptll, side="right") - 1
            for s in range(4):
                hist["ptll_reference"][s] += np.bincount(
                    b, weights=weights[:, s], minlength=len(config.DY_PT_EDGES) - 1
                )
            if entries % 2_000_000 < 200_000:
                print(
                    "GGF_HIST", region, record["sample"], entries, selected, flush=True
                )
    if not limit:
        reference = (
            Path(run)
            / era
            / "E/validation_candidate"
            / era
            / "raw"
            / region
            / (record["sample"] + ".npz")
        )
        with np.load(reference) as old:
            np.testing.assert_allclose(
                hist["event_yields"], old["event_yields"], rtol=2e-8, atol=1e-5
            )
            np.testing.assert_allclose(
                hist["ptll_reference"], old["ptll__sumw"], rtol=2e-8, atol=1e-5
            )
        assert selected == read(reference.with_suffix(".json"))["selected"]
    for j, name in enumerate(AXES):
        np.testing.assert_allclose(
            hist[name + "__sumw"].sum(axis=1),
            hist["population"][j, 0],
            rtol=1e-9,
            atol=1e-5,
        )
        np.testing.assert_allclose(
            hist["population"][j, 0] + hist["missing"][j],
            hist["event_yields"][:, 0],
            rtol=1e-9,
            atol=1e-5,
        )
    save(output, **hist)
    report = dict(
        provenance=provenance,
        region=region,
        sample=record["sample"],
        entries=entries,
        selected=selected,
        seconds=time.monotonic() - started,
        full_reference_check=not bool(limit),
        jet_truncation=jet_statistics,
        artifact_sha256=sha(output),
    )
    write(metadata, report)
    print("GGF_HIST_DONE", region, record["sample"], entries, selected, flush=True)
    return report


def render(run, era, region, records, limit):
    os.environ["RECOIL_RUN_DIR"] = str(run)
    import plots

    output = output_dir(run, era)
    region_records = [r for r in records if r["era"] == era and r["region"] == region]
    raw = {}
    for r in region_records:
        with np.load(output / "raw" / region / (r["sample"] + ".npz")) as handle:
            raw[r["sample"]] = dict(handle)
    data = raw["data"]
    alpha = float(
        data["event_yields"][0, 0]
        / sum(
            raw[r["sample"]]["event_yields"][0, 0]
            for r in region_records
            if r["sample"] != "data"
        )
    )
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Invalid regional normalization")
    if not limit:
        expected = read(
            Path(run)
            / era
            / "E/validation_candidate"
            / era
            / region
            / "normalization.json"
        )["alpha"]
        np.testing.assert_allclose(alpha, expected, rtol=2e-8)
    target = output / region
    target.mkdir(parents=True, exist_ok=True)
    rows, metrics = [], {}
    plot_values = {}
    with PdfPages(target / "summary.pdf") as summary:
        for j, (name, (edges, xlabel)) in enumerate(AXES.items()):
            groups = {}
            for r in region_records:
                if r["sample"] == "data":
                    continue
                group = groups.setdefault(
                    r["group"],
                    {
                        s: dict(
                            sumw=np.zeros(len(edges) - 1),
                            sumw2=np.zeros(len(edges) - 1),
                        )
                        for s in STAGES
                    },
                )
                for s, stage in enumerate(STAGES):
                    for moment, power in [("sumw", 1), ("sumw2", 2)]:
                        group[stage][moment] += (
                            raw[r["sample"]][name + "__" + moment][s] * alpha**power
                        )
            title = f"{era} {region.upper()} ggF DNN inputs; independent validation; SFs fitted in ZEE/WCR\nFour SF stages; statistical errors; finite tails folded into end bins"
            required = REQUIRED_JETS.get(name.removesuffix("__over_mass"), 0)
            if required:
                title += f"; Njet >= {required}"
            metrics[name] = dict(
                data_valid=float(data["population"][j, 0, 0]),
                data_missing=float(data["missing"][j, 0]),
                data_underflow=float(data["population"][j, 1, 0]),
                data_overflow=float(data["population"][j, 2, 0]),
                scales={},
            )
            for normalization in ("physical", "absolute", "shape"):
                plot_groups = (
                    plots.scaled_groups(groups, 1 / alpha)
                    if normalization == "physical"
                    else groups
                )
                scales = plots.agreement(
                    name,
                    edges,
                    data[name + "__sumw"][0],
                    data[name + "__sumw2"][0],
                    plot_groups,
                    target / (name + "_" + normalization),
                    title + "; " + normalization,
                    STAGES,
                    normalization,
                    xlabel,
                    summary,
                )
                if any(s is None for s in scales.values()):
                    raise ValueError(f"Cannot normalize {region}/{name}")
                metrics[name]["scales"][normalization] = scales
                metrics[name].setdefault("agreement", {})[normalization] = (
                    plots.agreement_metrics(
                        data[name + "__sumw"][0],
                        data[name + "__sumw2"][0],
                        plot_groups,
                        STAGES,
                        normalization,
                    )
                )
            plot_values[name + "__data"] = data[name + "__sumw"][0]
            plot_values[name + "__data_sumw2"] = data[name + "__sumw2"][0]
            plot_values[name + "__edges"] = edges
            for stage in STAGES:
                for moment in ("sumw", "sumw2"):
                    plot_values[name + "__" + stage + "__" + moment] = sum(
                        group[stage][moment] for group in groups.values()
                    )
            rows.append(
                f'<section id="{name}"><h2>{html.escape(name)}</h2><p>Valid data: {metrics[name]["data_valid"]:g}; missing/undefined: {metrics[name]["data_missing"]:g}</p>'
                + "".join(
                    f'<a href="{name}_{n}.pdf"><img loading="lazy" src="{name}_{n}.png" alt="{name} {n}"></a>'
                    for n in ("physical", "absolute", "shape")
                )
                + "</section>"
            )
    from runtime import save

    save(target / "plot_values.npz", **plot_values)
    from fit_statistics import finite

    write(
        target / "metrics.json",
        finite(dict(alpha=alpha, variables=metrics, stages=STAGES, max_entries=limit)),
    )
    (target / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>ggF DNN inputs '
        + region.upper()
        + "</title>"
        "<style>body{font-family:sans-serif;margin:24px}img{width:49%;max-width:900px}section{border-top:1px solid #ccc}h2{font-size:18px}</style>"
        f'<h1>{era} {region.upper()} ggF DNN inputs</h1><p><a href="summary.pdf">Summary PDF</a>; physical plots use luminosity × cross section; absolute plots use fixed nominal Data/MC normalization; shape plots normalize each stage to the valid data population.</p>'
        "<p>The original inclusive validation selection is retained. Missing-jet/undefined values are excluded from physical plots and counted below; finite tails are folded into endpoint bins. No model scaling or imputation is applied. pTMu2oMmm reproduces the original skim muon-1 numerator. The four __over_mass plots show mass-divided inputs.</p>"
        + "".join(rows)
    )
    print("GGF_RENDER_DONE", region, target, flush=True)


def aggregate(era):
    from runtime import RUN

    run = RUN
    records = read(run / "plotting_manifest.json")["records"]
    model = read(MODEL)
    if set(model["train_variables"]) != set(AXES) - {
        n + "__over_mass" for n in PTDECOR
    }:
        raise ValueError("Frozen ggF variable catalog differs from training metadata")
    for r in records:
        if r["era"] != era or r["region"] not in ("zcr", "srsb"):
            continue
        path = output_dir(run, era) / "raw" / r["region"] / (r["sample"] + ".npz")
        meta = read(path.with_suffix(".json"))
        if meta["provenance"]["record"] != r or meta["artifact_sha256"] != sha(path):
            raise ValueError(f"Stale ggF artifact: {path}")
    for region in ("zcr", "srsb"):
        render(run, era, region, records, 0)
    write(
        output_dir(run, era) / "manifest.json",
        dict(
            completed=True,
            era=era,
            variables=list(AXES),
            stages=STAGES,
            model_sha256=sha(MODEL),
            skim_sha256=sha(SKIM),
            normalizations=["physical", "absolute", "shape"],
            definitions=dict(
                pTMu2oMmm="Original training skim uses muon 1 numerator; preserved",
                source_year="Raw analysis year derived from explicit era, before any model encoding",
                missing="Undefined and missing-jet values excluded and counted; no imputation",
            ),
        ),
    )
