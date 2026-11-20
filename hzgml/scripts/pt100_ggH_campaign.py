#!/usr/bin/env python3
"""Produce corrected SRSB skims and a migration-aware ggH fit campaign."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np
import uproot

from dnn_fit_campaign import CONFIG, INPUTS, SOURCES, apply_command, resolve_parameters, run, write_json
from dnn_fit_output import mapping_record, systematic_plan
from dnn_root_io import open_published_root
from merge_dnn_fit_eras import merge_one

REPO = Path(__file__).resolve().parents[2]
SHARE = Path("/eos/user/j/jiehan/HMuMuShare")
WORKFLOW = REPO / "workflows/JetRecoilCorrectionWorkflow_forward_pt100_20261118"
FITS = SHARE / "JetRecoilCorrection_Run3/20261118_hehf_pt100"
EXTRA_CONFIG = "hzgml/data/apply_inputs_config_ggH.py"
EXTRA_INPUTS = "hzgml/data/inputs_config_higgs_extra_ggH.json"
TRAINER = "train_dnn_ggH_true_mass_decor.py"
PROXY = "/afs/cern.ch/user/j/jiehan/x509up_u152815"
FIT_FILES = ("fit_results.json", "covariance.npy", "covariance_sf.npy", "parameters.npy",
             "corrections.json", "corrections.json.gz", "corrections_metadata.json")
FOLD_SEED = 20250408


def read(path):
    return json.loads(Path(path).read_text())


def checksum(path):
    url = "root://eosuser.cern.ch"
    return subprocess.check_output(["xrdfs", url, "query", "checksum", str(path)], text=True).split()[1]


def load_skim(source):
    sys.path.insert(0, str(source / "recoil"))
    import skim_run3 as skim
    return skim


def configure_strict_boundary():
    import config
    import until

    original = until.jet_indices

    def jet_indices(pt, eta, matched):
        cells, keep = original(pt, eta, matched)
        # The +2.5 edge belongs to a fixed SF cell, as in the correctionlib export.
        keep_slice = eta == 2.5
        cells[keep_slice] -= 1
        return cells, keep

    until.jet_indices = jet_indices
    return config, until


def inspect_record(skim, record, paths, expected_fingerprint):
    if not paths:
        raise FileNotFoundError(f"Missing {record['era']}/{record['sample']}")
    era, is_data = record["era"], record["data"]
    record["inputs"] = [skim.inspect(path, era, is_data) for path in paths]
    for item in record["inputs"]:
        if item["production_fingerprint"] != expected_fingerprint:
            raise ValueError(f"Unexpected production fingerprint in {item['path']}")
        if item["entries"] <= 0:
            raise ValueError(f"Empty input {item['path']}")
    branches = set(record["inputs"][0]["branches"])
    if any(set(item["branches"]) != branches for item in record["inputs"]):
        raise ValueError(f"Mixed input schema for {record['era']}/{record['sample']}")
    record["shifts"] = sorted(k.split("__", 1)[1] for k in branches
                              if k.startswith("Jet_selectionMask__")) if record["signal"] else []
    if record["signal"] and len(record["shifts"]) != 24:
        raise ValueError(f"Expected 24 kinematic shifts for {record['era']}/{record['sample']}")
    denominator = None if is_data else sum(item["normalization_sumw"] for item in record["inputs"])
    record["normalization_sumw"] = denominator
    record["scale"] = 1.0 if is_data else record["lumi_pb"] * record["cross_section_pb"] / denominator
    return record


def prepare(run_dir, artifacts):
    if run_dir.exists() or artifacts.exists():
        raise FileExistsError("Use a new campaign run directory and EOS output directory")

    reference = runpy.run_path(str(REPO / CONFIG))["CONFIG"]
    config = deepcopy(runpy.run_path(str(REPO / EXTRA_CONFIG))["CONFIG"])
    years = [entry["year"] for entry in config["ggH"]["yearly_inputs"]]
    categories = list(read(REPO / INPUTS)["sample_list"])
    extra = list(read(REPO / EXTRA_INPUTS)["sample_list"])
    signals = set(reference["ggH"]["train_signal"]) | set(extra)
    if read(FITS / "fit_agreement_completion.json")["completed"] is not True:
        raise ValueError("Fit agreement stage is incomplete")

    source = run_dir / "source"
    logs = run_dir / "logs"
    logs.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    for name in SOURCES + ["hzgml/scripts/pt100_ggH_campaign.py", "hzgml/scripts/dnn_region_policy.py"]:
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, destination)
    recoil = source / "recoil"
    recoil.mkdir()
    for name in ("skim_run3.py", "skim_features.py", "skim_covariance.py", "eras.json", "model.json"):
        shutil.copy2(WORKFLOW / name, recoil / name)
    shutil.copytree(FITS / "code_snapshot/scripts", recoil / "scripts",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    plot_source = Path("scripts/Jet_RecoilCorrection/ggf_input_plots.py")
    shutil.copy2(WORKFLOW / plot_source, recoil / plot_source)
    skim = load_skim(source)
    catalog = read(WORKFLOW / "samples.json")["eras"]
    eras = read(recoil / "eras.json")
    known_xsecs = skim.literal_assignment(REPO / "hzgml/scripts/Jet_DNN/until.py", "XSECTIONS")
    native_signal_xsecs = dict(GluGluHToMuMu_M125="GluGluHto2Mu", VBFHToMuMu_M125="VBFHto2Mu",
                               ttHToMuMu_M125="TTH_Hto2Mu")

    tasks = []
    fit_hashes = {}
    for year_config in config["ggH"]["yearly_inputs"]:
        year = year_config["year"]
        source_fit = FITS / year / "E" / "fit"
        target_fit = artifacts / "corrections" / year
        target_fit.mkdir(parents=True)
        for name in FIT_FILES:
            shutil.copy2(source_fit / name, target_fit / name)
        fit, _ = skim.corrections(artifacts, year)
        if fit["jet_sf_fixed"] is None:
            raise ValueError(f"Missing fixed SF information for {year}")
        fit_hashes[year] = {name: skim.digest(target_fit / name) for name in FIT_FILES}
        write_json(artifacts / "corrections" / year / "source.json",
                   {"path": str(source_fit), "sha256": fit_hashes[year]})
        sample_catalog = {record["sample"]: record for record in catalog[year]["srsb"]}
        mc_fingerprints = {record["production_fingerprint"] for record in sample_catalog.values()
                           if record["sample"] != "data"}
        mc_fingerprint, = mc_fingerprints
        data_fingerprint = sample_catalog["data"]["production_fingerprint"]
        source_era = SHARE / f"SRSB_{year}"
        for category in categories + extra:
            sample = year_config["sample_map"][category]
            is_data = category == "data"
            paths = sorted((source_era / "data").glob("*.root")) if is_data else [source_era / "mc" / f"{sample}.root"]
            if is_data:
                xsec = None
            elif sample in sample_catalog:
                xsec = sample_catalog[sample]["cross_section_pb"]
            else:
                xsec = known_xsecs[native_signal_xsecs[category] if sample not in known_xsecs else sample]
            record = dict(index=len(tasks), era=year, region="srsb", category=category, sample=sample,
                          data=is_data, signal=category in signals, cross_section_pb=xsec,
                          lumi_pb=eras[year]["lumi_pb"], dy=category == "DY_105To160_ZpT-reweighted",
                          output=f"{year}/SRSB/{sample}.root")
            tasks.append((record, paths, data_fingerprint if is_data else mc_fingerprint))
        year_config["folder"] = str(artifacts / "roots" / year / "SRSB")
        year_config["tree_name"] = "Events"
    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(lambda task: inspect_record(skim, *task), tasks))
    print(f"Inspected {len(records)} SRSB sample inputs", flush=True)

    # Use one apply configuration with the complete sample map. The trainer still reads train_signal.
    (source / CONFIG).write_text("CONFIG = " + repr(config) + "\n")
    apply_inputs = read(source / INPUTS)
    apply_inputs["sample_list"].update(read(source / EXTRA_INPUTS)["sample_list"])
    write_json(source / INPUTS, apply_inputs)
    defaults = config["ggH"]["dnn_defaults"]
    resolved = resolve_parameters(source / "hzgml/scripts" / TRAINER, defaults, {})
    unsupported = sorted(set(defaults) - set(resolved))
    if any(resolved[k] != defaults[k] for k in defaults if k in resolved):
        raise ValueError("An effective trainer default differs from the template")
    writes = {str(p.relative_to(source)): skim.digest(p) for p in source.rglob("*") if p.is_file()}
    manifest = {"eras": years, "years": years, "records": records, "categories": categories + extra,
                "signals": sorted(signals), "source": str(source), "source_fit": str(FITS),
                "fit_sha256": fit_hashes, "sha256": writes,
                "template_sha256": skim.digest(REPO / CONFIG),
                "training_variables": config["ggH"]["train_variables"],
                "config": str(source / CONFIG), "inputs_config": str(source / INPUTS),
                "run_dir": str(run_dir), "artifacts": str(artifacts), "model_dir": str(artifacts / "models_dnn"),
                "training_script": TRAINER, "training_weights": {"reso": False, "reso_bkg": False},
                "defaults": {"folds": [0, 1, 2, 3], "seed": FOLD_SEED, "ptdecor": True, "optuna": False,
                             "dnn_defaults": defaults, "unsupported_template_keys": unsupported},
                "resolved_params": resolved}
    write_json(artifacts / "manifest.json", manifest)
    write_json(artifacts / "completion.json", {"completed": False, "stage": "prepared"})
    write_dag(manifest)


def write_dag(manifest):
    control = Path(manifest["run_dir"])
    artifacts = Path(manifest["artifacts"])
    worker = control / "worker.sh"
    worker.write_text(f"""#!/bin/bash
set -e -o pipefail
run="${{1:?AFS submission directory is required}}"
shift
source_dir="$run/source/hzgml"
source "$source_dir/scripts/setup_dnn_gpu.sh"
case "${{1:-}}" in
    inference-pilot|apply) export CUDA_VISIBLE_DEVICES=-1 ;;
esac
export X509_USER_PROXY="{PROXY}"
export PATH="$source_dir/scripts/root_tools:$PATH"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2
export CROWN_JET_PT_MODE=nominal
export MPLCONFIGDIR="${{_CONDOR_SCRATCH_DIR:-/tmp}}/dnn-mpl"
unset PYTHONHOME
exec python3 -B "$source_dir/scripts/pt100_ggH_campaign.py" worker "{artifacts / 'manifest.json'}" "$@"
""")
    worker.chmod(0o755)
    dag = ["NODE_STATUS_FILE pipeline.status 30 ALWAYS-UPDATE", "MAXJOBS SKIM 4", "MAXJOBS APPLY 4"]
    for name, memory, flavour in [("skim", 12000, "tomorrow"),
                                  ("check", 12000, "tomorrow"),
                                  ("train", 49152, "nextweek"),
                                  ("apply", 18000, "nextweek"),
                                  ("merge", 16000, "tomorrow")]:
        gpu_request = (
            "request_gpus = 1\nrequire_gpus = (GlobalMemoryMb >= 80000) && (Capability >= 7.0)\n"
            if name == "train" else ""
        )
        (control / f"{name}.sub").write_text(
            f"universe = vanilla\ninitialdir = {control}\nexecutable = $(initialdir)/worker.sh\n"
            f"arguments = $(initialdir) $(args)\nshould_transfer_files = NO\nx509userproxy = {PROXY}\n"
            f'request_cpus = 4\nrequest_memory = {memory}MB\nrequest_disk = 30000MB\n'
            f'+JobFlavour = "{flavour}"\nMY.WantOS = "el9"\n'
            'requirements = (TARGET.NODE_MOUNTS_AFS =?= true)\n'
            f"output = {control}/logs/$(node).$(ClusterId).out\n"
            f"error = {control}/logs/$(node).$(ClusterId).err\n"
            f"log = {control}/jobs.log\n{gpu_request}queue 1\n")

    def node(name, submit, args, category=None):
        dag.extend((f"JOB {name} {submit}.sub", f'VARS {name} node="{name}" args="{args}"',
                    f"RETRY {name} 1"))
        if category:
            dag.append(f"CATEGORY {name} {category}")
        return name

    pilots = [node(f"PILOT_{r['index']}", "skim", f"pilot {r['index']}", "SKIM") for r in pilot_records(manifest)]
    node("SKIM_PILOT_CHECK", "check", "pilot-check")
    dag.append(f"PARENT {' '.join(pilots)} CHILD SKIM_PILOT_CHECK")
    skims = [node(f"SKIM_{r['index']}", "skim", f"skim {r['index']}", "SKIM") for r in manifest["records"]]
    dag.append(f"PARENT SKIM_PILOT_CHECK CHILD {' '.join(skims)}")
    node("SKIM_CHECK", "check", "skim-check")
    dag.append(f"PARENT {' '.join(skims)} CHILD SKIM_CHECK")
    node("TRAIN", "train", "train")
    node("INFERENCE_PILOT", "apply", "inference-pilot")
    dag += ["PARENT SKIM_CHECK CHILD TRAIN", "PARENT TRAIN CHILD INFERENCE_PILOT"]
    apply = [node(f"APPLY_{y}", "apply", f"apply {y}", "APPLY") for y in manifest["years"]]
    dag.append(f"PARENT INFERENCE_PILOT CHILD {' '.join(apply)}")
    node("MERGE", "merge", "merge")
    dag.append(f"PARENT {' '.join(apply)} CHILD MERGE")
    (control / "campaign.dag").write_text("\n".join(dag) + "\n")


def pilot_records(manifest):
    return [r for r in manifest["records"]
            if r["category"] in {"data", "DY_105To160_ZpT-reweighted", *manifest["signals"]}
            and (r["era"] in ("2022", "2024") or r["category"] == "GluGluHToMuMu_M125")]


def verify_skim(manifest, skim, pilot=False):
    artifacts = Path(manifest["artifacts"])
    limit = 20000 if pilot else 0
    records = pilot_records(manifest) if pilot else manifest["records"]
    summaries = []
    for r in records:
        path = artifacts / ("pilot" if pilot else "roots") / r["output"]
        receipt = read(path.with_suffix(".json"))
        if (receipt["record"] != r or receipt["limit"] != limit
                or receipt["fit_sha256"] != manifest["fit_sha256"][r["era"]]["fit_results.json"]
                or checksum(path) != receipt["adler32"]):
            raise ValueError(f"Invalid skim receipt for {path}")
        if r["signal"]:
            plan = systematic_plan(receipt["branches"], manifest["training_variables"], {}, include_migrations=True)
            if {k for k, v in plan.items() if "category" in v} != set(r["shifts"]):
                raise ValueError(f"Shift coverage mismatch for {path}")
        checked = skim.validate_root(path, receipt["branches"], receipt["totals"])
        summaries.append({"era": r["era"], "category": r["category"], "path": str(path), **checked})
    write_json(artifacts / ("skim_pilot.json" if pilot else "skim_report.json"),
               {"passed": True, "files": summaries})


def check_boundaries(manifest, skim, config, until):
    import correctionlib

    artifacts = Path(manifest["artifacts"])
    for year in manifest["years"]:
        config.configure(year)
        fit, covariance = skim.corrections(artifacts, year)
        correction = correctionlib.CorrectionSet.from_file(
            str(artifacts / "corrections" / year / "corrections.json.gz"))
        pt = np.array([25., 99.99, 100., 99.99, 99.99, 99.99, 99.99, 100.])
        eta = np.array([3.0, 3.0, 3.0, -2.5, 2.5, -2.51, 2.51, 2.51])
        for label in (0, 1):
            cells, selected = until.jet_indices(pt, eta, np.full(len(pt), label))
            np.testing.assert_array_equal(selected, pt > 25.)
            expected = np.asarray(fit["jet_sf"]).ravel()[cells[selected]]
            np.testing.assert_allclose(expected, correction["jet_sf"].evaluate(
                np.full(selected.sum(), label), pt[selected], eta[selected]), rtol=1e-14)
            fixed = (abs(eta[selected]) <= 2.5) | (pt[selected] >= 100.)
            np.testing.assert_array_equal(expected[fixed], 1.)
            np.testing.assert_array_equal(covariance.diagonal()[cells[selected][fixed]], 0.)
        import awkward as ak
        empty = ak.unflatten(np.array([], dtype=np.int32), [0])
        np.testing.assert_array_equal(ak.prod(empty, axis=1), [1])
        np.testing.assert_array_equal(skim.sf_variance(empty, covariance), [0])


def train(manifest):
    artifacts = Path(manifest["artifacts"])
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    command = [sys.executable, Path(manifest["source"]) / "hzgml/scripts" / TRAINER, "-c", manifest["config"],
               "-r", "ggH", "-o", "models_dnn", "--save", "--seed", FOLD_SEED,
               "--train-cache-dir", artifacts / "train_cache"]
    if manifest["defaults"]["ptdecor"]:
        command.append("--ptdecor")
    if manifest.get("epoch_checkpoints"):
        checkpoint_root = artifacts / "checkpoints"
        checkpoint_root.mkdir(exist_ok=True)
        checkpoint_dir = tempfile.mkdtemp(prefix="attempt_", dir=checkpoint_root)
        print(f"Writing epoch checkpoints to {checkpoint_dir}", flush=True)
        command.extend(["--checkpoint-dir", checkpoint_dir])
    run(command, artifacts)
    model_dir = Path(manifest["model_dir"])
    hashes = {}
    from skim_run3 import digest
    for fold in range(4):
        for name in (f"DNN_ggH_{fold}.keras", f"DNN_scaler_ggH_{fold}.pkl", f"DNN_metadata_ggH_{fold}.json"):
            hashes[name] = digest(model_dir / name)
        with zipfile.ZipFile(model_dir / f"DNN_ggH_{fold}.keras") as model:
            saved = json.loads(model.read("config.json"))["config"]["hparams"]
        for key in set(manifest["defaults"]["dnn_defaults"]) & set(manifest["resolved_params"]):
            if saved.get(key) != manifest["resolved_params"][key]:
                raise ValueError(f"Saved model parameter mismatch for fold {fold}: {key}")
    write_json(artifacts / "training_provenance.json",
               {"model_sha256": hashes, "resolved_params": manifest["resolved_params"]})


def apply(manifest, year, pilot=False):
    artifacts = Path(manifest["artifacts"])
    base = artifacts / ("inference_pilot" if pilot else "outputs")
    transformers = artifacts / ("inference_pilot_transformers" if pilot else "transformers") / year
    # Keep full nominal training signal input for the pilot transformers.
    extras = ["--max-chunks", "1", "-cat", "GluGluHToMuMu_M125"] if pilot else []
    run(apply_command(manifest, year, base / year, "--output-profile", "fit", "--systematics",
                      "--include-jet-migrations", "--disable-2223-jet-patch", "--chunk-size",
                      "20000" if pilot else "100000", "--save-score-transformers", transformers, *extras), artifacts)


def validate_apply(manifest, year, category, *, pilot=False):
    artifacts = Path(manifest["artifacts"])
    r = next(r for r in manifest["records"] if r["era"] == year and r["category"] == category)
    output = artifacts / ("inference_pilot" if pilot else "outputs") / year / "ggH" / f"{category}.root"
    summary = read(str(output) + ".summary.json")
    source_path = Path(manifest.get("skim_root", artifacts / "roots")) / r["output"]
    if summary["inputs"] != [str(source_path)]:
        raise ValueError(f"Input identity mismatch for {output}")
    weight_moments = {}
    migration = {}
    offset = 0
    with (uproot.open(source_path) as source, open_published_root(output) as scored):
        src = source["Events"]
        dest = scored["test"]
        plan = (systematic_plan(src.keys(), manifest["training_variables"], {},
                                allow_nominal_only=not r["signal"], include_migrations=r["signal"])
                if not r["data"] else {})
        if mapping_record(plan) != summary["systematics"] or dest.keys() != summary["branches"]:
            raise ValueError(f"Unexpected fit schema for {output}")
        ident = ["run", "lumi", "event", "diMufsr_kit_BSC_mass"]
        inputs = set(ident) | {"cate_index", "eventWeight"}
        for var in plan.values():
            inputs.add(var["weight"])
            if "category" in var:
                inputs.add(var["category"])
        for raw in src.iterate(sorted(inputs), step_size=100000, entry_stop=20000 if pilot else None, library="np"):
            in_mass = (raw[ident[-1]] >= 110.) & (raw[ident[-1]] <= 150.)
            nominal = in_mass & (raw["cate_index"] == 4)
            masks = {}
            union = nominal.copy()
            for name, var in plan.items():
                if "category" in var:
                    shifted = in_mass & (raw[var["category"]] == 4)
                    masks[name] = shifted
                    union |= shifted
                    counts = migration.setdefault(name, [0, 0, 0])
                    counts[0] += int(np.count_nonzero(shifted & ~nominal))
                    counts[1] += int(np.count_nonzero(nominal & ~shifted))
                    counts[2] += int(np.count_nonzero(shifted & nominal))
            size = int(np.count_nonzero(union))
            actual = dest.arrays(entry_start=offset, entry_stop=offset + size, library="np")
            offset += size
            for name in ident:
                np.testing.assert_array_equal(actual[name], raw[name][union])
            np.testing.assert_array_equal(actual["source_year"], np.full(size, manifest["years"].index(year)))
            np.testing.assert_array_equal(actual["score_t"], actual["dnn_score_t"])
            expected = {"eventWeight": np.where(nominal, raw["eventWeight"], 0.)[union]}
            for name, var in plan.items():
                suffix = var["suffix"]
                expected[f"eventWeight_{suffix}"] = np.where(
                    masks.get(name, nominal), raw[var["weight"]], 0.)[union]
                if not var["features"]:
                    np.testing.assert_array_equal(actual[f"score_t_{suffix}"], actual["score_t"])
            for name, weight in expected.items():
                np.testing.assert_array_equal(actual[name], weight)
                moments = weight_moments.setdefault(name, [0.0, 0.0])
                moments[0] += float(np.sum(weight, dtype=np.float64))
                moments[1] += float(np.sum(np.square(weight), dtype=np.float64))
            for name, values in actual.items():
                if name.startswith("score_t") and not np.all(np.isfinite(values) & (values >= 0.) & (values <= 1.)):
                    raise ValueError(f"Invalid score in {output}:{name}")
        if offset != dest.num_entries or offset != summary["entries"]:
            raise ValueError(f"Fit row count mismatch for {output}")
    for name, (sumw, _) in weight_moments.items():
        if not math.isclose(sumw, summary["sumw"][name], rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError(f"Fit weight sum mismatch for {output}:{name}")
    return {"era": year, "category": category, "entries": offset, "sumw_sumw2": weight_moments,
            "migrations": {k: dict(zip(("in", "out", "both"), v)) for k, v in migration.items()}}


def merge(manifest, *, mark_complete=True):
    artifacts = Path(manifest["artifacts"])
    provenance = read(artifacts / "training_provenance.json")
    checks = [item for year in manifest["years"] for item in read(artifacts / "outputs" / year / "validation.json")]
    if len(checks) != len(manifest["records"]):
        raise ValueError("Incomplete per-era validation")
    written = []
    by_category = {}
    for check in checks:
        by_category.setdefault(check["category"], []).append(check)
    for category in manifest["categories"]:
        with tempfile.TemporaryDirectory(prefix="pt100_merge_", dir=os.environ.get("_CONDOR_SCRATCH_DIR")) as scratch:
            result = merge_one(artifacts, manifest["years"], category, scratch, provenance)
        if result["entries"] != sum(c["entries"] for c in by_category[category]):
            raise ValueError(f"Merged count mismatch: {category}")
        for name, actual in result["sumw"].items():
            expected = sum(c["sumw_sumw2"].get(name, c["sumw_sumw2"]["eventWeight"])[0] for c in by_category[category])
            if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-7):
                raise ValueError(f"Merged yield mismatch: {category}:{name}")
        written.append(result)
        print(f"Validated merged {category}: {result['entries']} entries", flush=True)
    write_json(artifacts / "report.json",
               {"complete": True, "years": manifest["years"], "samples": written, "per_era": checks})
    if mark_complete:
        write_json(artifacts / "completion.json", {"completed": True, "stage": "validated",
                   "skim_files": len(manifest["records"]), "per_era_fit_files": len(checks), "merged_files": len(written)})


def worker(manifest_path, stage, target):
    manifest = read(manifest_path)
    artifacts = Path(manifest["artifacts"])
    source = Path(manifest["source"])
    skim = load_skim(source)
    if stage in {"skim", "pilot", "pilot-check"}:
        config, until = configure_strict_boundary()
    if stage in {"skim", "pilot"}:
        r = manifest["records"][int(target)]
        for name in ("fit_results.json", "covariance.npy"):
            if skim.digest(artifacts / "corrections" / r["era"] / name) != manifest["fit_sha256"][r["era"]][name]:
                raise ValueError(f"Correction changed for {r['era']}: {name}")
        skim.worker(artifacts, int(target), limit=20000 if stage == "pilot" else 0)
    elif stage == "pilot-check":
        for path, sha in manifest["sha256"].items():
            if skim.digest(source / path) != sha:
                raise ValueError(f"Source snapshot changed: {path}")
        check_boundaries(manifest, skim, config, until)
        verify_skim(manifest, skim, pilot=True)
    elif stage == "skim-check":
        verify_skim(manifest, skim)
    elif stage == "train":
        train(manifest)
        if manifest.get("epoch_checkpoints"):
            write_json(artifacts / "completion.json", {"completed": True, "stage": "trained"})
    elif stage == "inference-pilot":
        checks = []
        for year in ("2022", "2024"):
            apply(manifest, year, pilot=True)
            checks.append(validate_apply(manifest, year, "GluGluHToMuMu_M125", pilot=True))
        if not all(sum(v[flow] for check in checks for v in check["migrations"].values()) > 0
                   for flow in ("in", "out", "both")):
            raise ValueError("Inference pilot did not observe both migration directions and stable events")
        write_json(artifacts / "inference_pilot.json", {"passed": True, "checks": checks})
    elif stage == "apply":
        apply(manifest, target)
        checks = [validate_apply(manifest, target, c) for c in manifest["categories"]]
        write_json(artifacts / "outputs" / target / "validation.json", checks)
    elif stage == "merge":
        merge(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--tag", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    job = commands.add_parser("worker")
    job.add_argument("manifest", type=Path)
    job.add_argument("stage", choices=("pilot", "pilot-check", "skim", "skim-check", "train",
                                       "inference-pilot", "apply", "merge"))
    job.add_argument("target", nargs="?")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(REPO / "condor/dnn/pt100" / args.tag, SHARE / "DNNJetPt100_Run3" / args.tag)
    else:
        worker(args.manifest, args.stage, args.target)


if __name__ == "__main__":
    main()
