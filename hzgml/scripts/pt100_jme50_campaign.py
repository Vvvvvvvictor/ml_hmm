#!/usr/bin/env python3
"""Prepare one DAG for era-dependent JME veto correction, skim and GRL training."""
import argparse
import os
from pathlib import Path
import runpy
import shutil
import sys

import numpy as np
import uproot

import pt100_grl_campaign as grl
from pt100_ggH_campaign import FIT_FILES, SHARE, WORKFLOW, load_skim, pilot_records, verify_skim
from pt100_grl_campaign import BASELINE, CONTROL, digest, read, run, write_json

TAG = "20261120_muon2_bsc_jme50_grl"
INCLUSIVE = BASELINE.parent / "20261120_muon2_bsc_grl_lam0p2_tau5"
PYTHON = "/eos/user/j/jiehan/ml-gpu-env/bin/python3"


def selection(era):
    return "jme" if int(era[:4]) <= 2023 else "he50"


def substitute(path, old, new):
    source = path.read_text()
    if source.count(old) != 1:
        raise ValueError(f"Expected one replacement in {path}: {old!r}")
    path.write_text(source.replace(old, new))


def prepare_recoil_core(destination):
    shutil.copytree(WORKFLOW, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
    scripts = destination / "scripts/Jet_RecoilCorrection"
    substitute(destination / "run.py", '            "inclusive",\n',
               '            "jme" if int(era[:4]) <= 2023 else "he50",\n')
    substitute(scripts / "config.py", '    if jet_selection != "inclusive":\n',
               '    if jet_selection not in {"inclusive", "jme", "he50"}:\n')
    substitute(scripts / "until.py", '    return keep\n\n\ndef bin_indices',
               '''    abs_eta = np.abs(eta)
    if config.JET_SELECTION == "jme":
        keep &= ~((abs_eta > 2.5) & (pt < 50.0))
    elif config.JET_SELECTION == "he50":
        keep &= ~((abs_eta > 2.5) & (abs_eta < 3.0) & (pt < 50.0))
    return keep


def bin_indices''')
    substitute(scripts / "until.py", '    valid = pt_valid & eta_valid & selected_jet_mask(pt, eta)\n',
               '''    if config.JET_SELECTION in {"jme", "he50"}:
        eta_idx[eta == 2.5] -= 1
    if config.JET_SELECTION == "he50":
        eta_idx[eta == -3.0] -= 1
    valid = pt_valid & eta_valid & selected_jet_mask(pt, eta)
''')
    export = scripts / "sf_export.py"
    substitute(export, '    expected_fixed = np.broadcast_to(~eligible, jet_shape)\n',
               '''    selector = manifest["configuration"]["selection"]["jet_selection"]
    if selector in {"he50", "jme"}:
        eta_low = np.minimum(abs(eta_grid[:-1]), abs(eta_grid[1:]))
        eta_high = np.maximum(abs(eta_grid[:-1]), abs(eta_grid[1:]))
        veto = (pt_grid[1:, None] <= 50) & (eta_low[None, :] >= 2.5)
        if selector == "he50":
            veto &= eta_high[None, :] <= 3.0
        eligible &= ~veto
    expected_fixed = np.broadcast_to(~eligible, jet_shape)
''')
    substitute(export, '    corrections = []\n',
               '''    if selector == "he50":
        hf_boundary = export_eta.index(-3.0)
        export_eta.insert(hf_boundary + 1, float(np.nextafter(-3.0, np.inf)))
    corrections = []
''')
    substitute(export, '        jet_values = np.insert(jet_values, boundary_bin, boundary_value, axis=2)\n',
               '''        jet_values = np.insert(jet_values, boundary_bin, boundary_value, axis=2)
        if selector == "he50":
            jet_values = np.insert(jet_values, hf_boundary, jet_values[:, :, hf_boundary - 1], axis=2)
''')
    substitute(destination / "skim_run3.py", "    config.configure(r['era']);fit,cov=corrections(run,r['era'])\n",
               "    config.configure(r['era'], manifest['jet_selections'][r['era']]);fit,cov=corrections(run,r['era'])\n")
    model = read(destination / "model.json")
    model["reference_changes"].append("pre-fit and skim: early eras JME HE/HF veto, late eras HE-only veto")
    write_json(destination / "model.json", model)


def prepare(tag):
    inclusive = read(INCLUSIVE / "manifest.json")
    if any(inclusive["resolved_params"].get(k) != v for k, v in grl.GRL_PARAMETERS.items()):
        raise ValueError("GRL defaults differ from the inclusive reference")
    grl.configure(tag)
    control = CONTROL / tag
    artifacts = BASELINE.parent / tag
    correction = SHARE / "JetRecoilCorrection_Run3" / tag
    core = control / "recoil_core"
    prepare_recoil_core(core)
    recoil_control = control / "recoil_condor"
    run([PYTHON, "-B", core / "run.py", "--prepare-only", "--output", correction,
         "--control-dir", recoil_control], control)

    source = control / "source"
    recoil = source / "recoil"
    recoil.mkdir()
    for filename in ("skim_run3.py", "skim_features.py", "skim_covariance.py", "eras.json", "model.json"):
        shutil.copy2(core / filename, recoil / filename)
    shutil.copytree(core / "scripts", recoil / "scripts")
    coordinator = source / "hzgml/scripts" / Path(__file__).name
    shutil.copy2(__file__, coordinator)
    manifest_path = artifacts / "manifest.json"
    manifest = read(manifest_path)
    if manifest["resolved_params"] != inclusive["resolved_params"]:
        raise ValueError("Train parameters differ from inclusive GRL")
    cfg = Path(manifest["config"])
    data = runpy.run_path(str(cfg))["CONFIG"]
    for year in data["ggH"]["yearly_inputs"]:
        year["folder"] = str(artifacts / "roots" / year["year"] / "SRSB")
    cfg.write_text("CONFIG = " + repr(data) + "\n")
    manifest.update(
        baseline=str(INCLUSIVE), skim_root=str(artifacts / "roots"), source_fit=str(correction),
        fit_sha256={}, defer_comparison=True, jet_selections={y:selection(y) for y in manifest["years"]},
        sha256={str(f.relative_to(source)): digest(f) for f in source.rglob("*") if f.is_file()},
    )
    write_json(manifest_path, manifest)
    write_pipeline(control, recoil_control, manifest)
    return control


def write_pipeline(control, recoil_control, manifest):
    worker = control / "cpu_worker.sh"
    template = (control / "worker.sh").read_text()
    worker.write_text(template[:template.index("exec python3")] +
                      f'exec python3 -B "$source_dir/scripts/{Path(__file__).name}" worker "{Path(manifest["artifacts"]) / "manifest.json"}" "$2" "${{3:-}}"\n')
    worker.chmod(0o755)
    submit = (control / "train_apply.sub").read_text()
    cpu = submit.replace("/worker.sh", "/cpu_worker.sh").replace(
        "arguments = $(initialdir)", "arguments = $(initialdir) $(stage) $(index)")
    cpu = "\n".join(line for line in cpu.splitlines()
                    if not line.startswith(("request_gpus", "require_gpus")))
    cpu = cpu.replace("request_memory = 15GB", "request_memory = 16GB")
    cpu = cpu.replace("/logs/train_apply.", "/logs/$(node).").replace(
        '"pt100_muon2_bsc_grl_train_apply"', '"pt100_jme50_cpu"')
    (control / "cpu.sub").write_text(cpu + "\n")
    (control / "train_apply.sub").write_text(submit.replace(
        '"pt100_muon2_bsc_grl_train_apply"', '"pt100_jme50_grl_train_apply"'))
    lines = [
        "NODE_STATUS_FILE pipeline.status 30 ALWAYS-UPDATE",
        f"SUBDAG EXTERNAL RECOIL {recoil_control / 'pipeline.dag'} DIR {recoil_control}",
        "MAXJOBS SKIM 4",
    ]

    def node(name, stage, index="-", category=None):
        lines.extend((f"JOB {name} cpu.sub", f'VARS {name} node="{name}" stage="{stage}" index="{index}"',
                      f"RETRY {name} 1"))
        if category:
            lines.append(f"CATEGORY {name} {category}")
        return name

    def dependency(parents, children):
        lines.append(f"PARENT {' '.join(parents)} CHILD {' '.join(children)}")

    publish = node("PUBLISH", "publish")
    pilots = [node(f"PILOT_{r['index']}", "pilot", r["index"], "SKIM") for r in pilot_records(manifest)]
    check_pilots = node("CHECK_PILOTS", "pilot-check")
    skims = [node(f"SKIM_{r['index']}", "skim", r["index"], "SKIM") for r in manifest["records"]]
    check_skims = node("CHECK_SKIMS", "skim-check")
    lines.extend(("JOB TRAIN_APPLY train_apply.sub", "RETRY TRAIN_APPLY 1"))
    compare = node("COMPARE", "compare")
    dependency(["RECOIL"], [publish])
    dependency([publish], pilots)
    dependency(pilots, [check_pilots])
    dependency([check_pilots], skims)
    dependency(skims, [check_skims])
    dependency([check_skims], ["TRAIN_APPLY"])
    dependency(["TRAIN_APPLY"], [compare])
    (control / "pipeline.dag").write_text("\n".join(lines) + "\n")


def check_selection(years):
    import config
    from until import selected_jets, selected_jet_mask
    import awkward as ak

    for year in years:
        selector = selection(year)
        config.configure(year, selector)
        abs_eta = np.array([2.5, 2.5001, 2.9999, 3.0, 3.001, 4.7])
        expected = np.array([True, False, False, selector == "he50", selector == "he50", False])
        for sign in (-1, 1):
            eta = sign * abs_eta
            np.testing.assert_array_equal(selected_jet_mask(np.full(6,49.99), eta), expected)
            np.testing.assert_array_equal(selected_jet_mask(np.full(6,50.0), eta), abs_eta < 4.7)
            np.testing.assert_array_equal(selected_jet_mask(np.full(6,25.0), eta), np.zeros(6,bool))
        for is_data in (False, True):
            arrays = ak.Array({"Jet_corrEta": [[], [2.8]], "Jet_corrPt": [[], [49.99]],
                               "Jet_corrMass": [[], [5.]], "Jet_corrPhi": [[], [0.]],
                               "Jet_isGenMatched": [[], [1]]})
            np.testing.assert_array_equal(ak.num(selected_jets(arrays, is_data)[0]), [0,0])
            shifted = ak.with_field(arrays, ak.Array([[], [50.01]]), "Jet_corrPt")
            np.testing.assert_array_equal(ak.num(selected_jets(shifted, is_data)[0]), [0,1])
    print("JME boundaries, empty events and shifted threshold migrations validated", flush=True)


def publish(manifest_path, manifest, skim, recoil_snapshot=None):
    import config
    import correctionlib
    from until import jet_indices

    fit_root = Path(manifest["source_fit"])
    artifacts = Path(manifest["artifacts"])
    if recoil_snapshot is None:
        if read(fit_root / "completion.json")["completed"] is not True:
            raise ValueError("Correction workflow did not complete")
    else:
        # Validate the fit and exported corrections independently of plot generation.
        exporter = recoil_snapshot / "scripts/Jet_RecoilCorrection/sf_export.py"
        for year in manifest["years"]:
            run([sys.executable, "-B", exporter, "--run-dir", fit_root / year / "E",
                 "--verify-only"], artifacts)
    check_selection(manifest["years"])
    hashes = {}
    for year, selector in manifest["jet_selections"].items():
        source = fit_root / year / "E/fit"
        target = artifacts / "corrections" / year
        target.mkdir(parents=True, exist_ok=True)
        for name in FIT_FILES:
            shutil.copy2(source / name, target / name)
        fit, covariance = skim.corrections(artifacts, year)
        exported = read(target / "corrections_metadata.json")
        if fit["jet_selection"] != selector or exported["selection"]["jet_selection"] != selector:
            raise ValueError(f"Wrong correction selection for {year}")
        config.configure(year, selector)
        payload = correctionlib.CorrectionSet.from_file(str(target / "corrections.json.gz"))
        pt, eta = (v.ravel() for v in np.meshgrid([26.,49.99,50.,99.99,100.],
                                                 [-3.001,-3.,-2.999,-2.5,2.5,2.501,2.999,3.,3.001]))
        sf = np.asarray(fit["jet_sf"]).ravel()
        fixed = np.asarray(fit["jet_sf_fixed"]).ravel()
        np.testing.assert_array_equal(sf[fixed], 1.)
        lo, hi = fit["parameter_slices"]["jet"]
        np.testing.assert_array_equal(covariance.diagonal()[lo:hi][fixed], 0.)
        for label in (0, 1):
            cells, keep = jet_indices(pt, eta, np.full(len(pt),label))
            np.testing.assert_allclose(sf[cells[keep]], payload["jet_sf"].evaluate(
                np.full(int(keep.sum()), label), pt[keep], eta[keep]), rtol=1e-14)
        hashes[year] = {name:digest(target / name) for name in FIT_FILES}
        write_json(target / "source.json", {"path":str(source), "sha256":hashes[year], "jet_selection":selector})
    manifest["fit_sha256"] = hashes
    write_json(manifest_path, manifest)


def skim_yields(manifest_path, root):
    from dnn_root_io import open_published_root

    manifest = read(manifest_path)
    results = {}
    for r in manifest["records"]:
        counts = np.zeros((2, 6),dtype=np.int64)
        weights = np.zeros_like(counts,dtype=float)
        with open_published_root(root / r["output"]) as f:
            for a in f["Events"].iterate(["cate_index","njets","eventWeight"],library="np",step_size="32 MB"):
                keep = np.isin(a["cate_index"], [3,4])
                bins = (a["cate_index"][keep].astype(int)-3)*6+np.minimum(a["njets"][keep].astype(int),5)
                counts += np.bincount(bins,minlength=12).reshape(2,6)
                weights += np.bincount(bins,weights=a["eventWeight"][keep],minlength=12).reshape(2,6)
        results.setdefault(r["era"],{})[r["category"]] = {"counts":counts.tolist(),"sumw":weights.tolist()}
    results["Run 3"] = {
        category: {
            key: np.sum([results[year][category][key] for year in manifest["years"]], axis=0).tolist()
            for key in ("counts", "sumw")
        }
        for category in manifest["categories"]
    }
    return results


def compare(manifest_path, manifest):
    artifacts = Path(manifest["artifacts"])
    inclusive = Path(manifest["baseline"])
    status = read(artifacts / "completion.json")
    if not status["completed"] or not read(inclusive / "completion.json")["completed"]:
        raise ValueError("Both GRL Apply campaigns must finish before comparison")
    run([sys.executable, Path(__file__).with_name("pt100_grl_comparison.py"), manifest_path], artifacts)
    write_json(artifacts / "jet_category_comparison.json", {
        "categories":[3,4], "jet_bins":[0,1,2,3,4,"5+"],
        "inclusive":skim_yields(inclusive / "manifest.json", Path(read(inclusive / "manifest.json")["skim_root"])),
        "jme_veto":skim_yields(manifest_path, Path(manifest["skim_root"])),
    })
    write_json(artifacts / "completion.json", {**status, "comparison":"grl_comparison.json",
                                             "jet_category_comparison":"jet_category_comparison.json"})


def worker(manifest_path, stage, index, recoil_snapshot=None):
    manifest = read(manifest_path)
    if stage == "compare":
        compare(manifest_path, manifest)
        return
    artifacts = Path(manifest["artifacts"])
    skim = load_skim(Path(manifest["source"]))
    if stage == "publish":
        source = Path(manifest["source"])
        for path, sha in manifest["sha256"].items():
            if digest(source / path) != sha:
                raise ValueError(f"Source snapshot changed: {path}")
        publish(manifest_path, manifest, skim, recoil_snapshot)
    elif stage in {"pilot","skim"}:
        record = manifest["records"][int(index)]
        for name, sha in manifest["fit_sha256"][record["era"]].items():
            if digest(artifacts / "corrections" / record["era"] / name) != sha:
                raise ValueError(f"Skim correction changed for {record['era']}/{name}")
        skim.worker(artifacts, int(index), limit=20000 if stage == "pilot" else 0)
    else:
        verify_skim(manifest, skim, pilot=stage == "pilot-check")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command",required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--tag",default=TAG)
    stage = sub.add_parser("worker")
    stage.add_argument("manifest",type=Path)
    stage.add_argument("stage",choices=("publish","pilot","pilot-check","skim","skim-check","compare"))
    stage.add_argument("index")
    stage.add_argument("--recoil-snapshot", type=Path,
                       help="Verify and publish exported fits independently of recoil plots")
    a = p.parse_args()
    if a.command == "prepare":
        print(prepare(a.tag))
    else:
        worker(a.manifest,a.stage,a.index,a.recoil_snapshot)


if __name__ == "__main__":
    main()
