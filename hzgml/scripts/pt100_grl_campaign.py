#!/usr/bin/env python3
"""Train and apply the Run 3 pt100 GRL model in one Condor job."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tempfile
import zipfile

from dnn_fit_campaign import SOURCES
from pt100_ggH_campaign import (
    PROXY, REPO, SHARE, apply, merge, read, resolve_parameters, run, validate_apply, write_json,
)

BASELINE = SHARE / "DNNJetPt100_Run3/20261118_muon2_bsc"
CONTROL = REPO / "condor/dnn/pt100"
GRL_PARAMETERS = {
    "domain_objective": "grl",
    "domain_lambda": 0.2,
    "domain_lambda_warmup_epochs": 3,
    "domain_lambda_schedule": "warmup_then_exp_tau5",
}
FOLD_PRODUCTS = (
    "DNN_ggH_{fold}.keras",
    "DNN_scaler_ggH_{fold}.pkl",
    "DNN_tsf_ggH_{fold}.pkl",
    "DNN_metadata_ggH_{fold}.json",
    "DNN_history_ggH_{fold}.json",
    "DNN_metrics_ggH_{fold}.json",
)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def configure(tag):
    run_dir = CONTROL / tag
    artifacts = BASELINE.parent / tag
    if run_dir.exists() or artifacts.exists():
        raise FileExistsError(f"Choose an unused tag: {tag}")
    baseline = read(BASELINE / "manifest.json")
    if not read(BASELINE / "completion.json").get("merged_files") == len(baseline["categories"]):
        raise ValueError("Baseline validation is incomplete")
    roots = BASELINE / "roots"
    for record in baseline["records"]:
        path = roots / record["output"]
        if not path.is_file():
            raise FileNotFoundError(path)

    source = run_dir / "source"
    shutil.copytree(Path(baseline["source"]) / "hzgml", source / "hzgml",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    scripts = [name for name in SOURCES if name.startswith("hzgml/scripts/")]
    scripts += [f"hzgml/scripts/{name}" for name in (
        "dnn_region_policy.py", "pt100_ggH_campaign.py", "pt100_grl_campaign.py", "pt100_grl_comparison.py",
    )]
    for name in scripts:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, target)
    config_path = source / "hzgml/data" / Path(baseline["config"]).name
    configuration = runpy.run_path(baseline["config"])["CONFIG"]
    defaults = configuration["ggH"]["dnn_defaults"]
    defaults.update(GRL_PARAMETERS)
    resolved = resolve_parameters(source / "hzgml/scripts" / baseline["training_script"], defaults, GRL_PARAMETERS)
    for year in configuration["ggH"]["yearly_inputs"]:
        if Path(year["folder"]) != roots / year["year"] / "SRSB":
            raise ValueError(f"Baseline input path mismatch for {year['year']}")
    config_path.write_text("CONFIG = " + repr(configuration) + "\n")

    artifacts.mkdir()
    (run_dir / "logs").mkdir()
    manifest = dict(baseline)
    manifest.update(
        baseline=str(BASELINE), run_dir=str(run_dir), artifacts=str(artifacts),
        source=str(source), skim_root=str(roots), config=str(config_path),
        inputs_config=str(source / "hzgml/data" / Path(baseline["inputs_config"]).name),
        model_dir=str(artifacts / "models_dnn"), epoch_checkpoints=False,
        sha256={str(path.relative_to(source)): digest(path)
                for path in sorted(source.rglob("*")) if path.is_file()},
    )
    manifest["defaults"]["dnn_defaults"] = defaults
    manifest["resolved_params"] = resolved
    write_json(artifacts / "manifest.json", manifest)
    write_json(artifacts / "completion.json", {"completed": False, "stage": "prepared"})

    worker = run_dir / "worker.sh"
    worker.write_text(f"""#!/bin/bash
set -eo pipefail
run="${{1:?AFS submission directory is required}}"
source_dir="$run/source/hzgml"
source "$source_dir/scripts/setup_dnn_gpu.sh"
: "${{_CONDOR_SCRATCH_DIR:?Condor scratch directory is required}}"
export X509_USER_PROXY="{PROXY}"
export PATH="$source_dir/scripts/root_tools:$PATH"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2 TF_FORCE_GPU_ALLOW_GROWTH=true
export CROWN_JET_PT_MODE=nominal MPLBACKEND=Agg
export MPLCONFIGDIR="$_CONDOR_SCRATCH_DIR/matplotlib"
unset PYTHONHOME
exec python3 -B "$source_dir/scripts/pt100_grl_campaign.py" worker "{artifacts / 'manifest.json'}"
""")
    worker.chmod(0o755)
    (run_dir / "train_apply.sub").write_text(f"""universe = vanilla
notification = Never
initialdir = {run_dir}
executable = $(initialdir)/worker.sh
arguments = $(initialdir)
output = $(initialdir)/logs/train_apply.$(ClusterId).out
error = $(initialdir)/logs/train_apply.$(ClusterId).err
log = $(initialdir)/logs/train_apply.$(ClusterId).log
should_transfer_files = NO
x509userproxy = {PROXY}
request_cpus = 4
request_memory = 15GB
request_disk = 64GB
request_gpus = 1
require_gpus = (GlobalMemoryMb >= 40000) && (Capability >= 8.0)
requirements = (TARGET.NODE_MOUNTS_AFS =?= true)
MY.WantOS = "el9"
+JobFlavour = "nextweek"
+JobBatchName = "pt100_muon2_bsc_grl_train_apply"
queue 1
""")
    print(run_dir / "train_apply.sub")


def log_resources(label, scratch):
    disk = shutil.disk_usage(scratch)
    group = Path("/sys/fs/cgroup") / Path(*Path(
        next(line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines()
             if line.startswith("0::"))
    ).parts[1:]) if Path("/sys/fs/cgroup/cgroup.controllers").exists() else None
    memory = {}
    if group:
        for key in ("memory.current", "memory.peak", "memory.max"):
            resource = group / key
            if resource.is_file():
                memory[key] = resource.read_text().strip()
    print(f"PT100 resources {label}: disk_used={disk.used} disk_free={disk.free} memory={memory}", flush=True)


def products(directory, names):
    return {name: digest(directory / name) for name in names}


def output_size(path):
    path = Path(os.path.normpath(path))
    try:
        return path.stat().st_size
    except FileNotFoundError:
        if path.suffix != ".root" or not str(path).startswith("/eos/"):
            raise
        # The EOS FUSE mount can lag the XRootD namespace after publication.
        result = subprocess.run(["xrdfs", "root://eosuser.cern.ch", "stat", str(path)],
                                capture_output=True, text=True, check=False)
        if result.returncode:
            raise
        for line in result.stdout.splitlines():
            if line.startswith("Size:"):
                return int(line.split()[1])
        raise ValueError(f"EOS stat did not return a size for {path}")


def has_receipt(path, lineage, directory, files):
    if not path.is_file():
        return False
    receipt = read(path)
    if receipt["lineage"] != lineage:
        return False
    try:
        return receipt["files"] == {name: output_size(directory / name) for name in files}
    except FileNotFoundError:
        return False


def save_receipt(path, lineage, directory, files):
    write_json(path, {"lineage": lineage, "files": {name: output_size(directory / name) for name in files}})


def check_model(manifest, models, fold):
    model_path = models / f"DNN_ggH_{fold}.keras"
    with zipfile.ZipFile(model_path) as z:
        saved = json.loads(z.read("config.json"))["config"]["hparams"]
    expected = manifest["resolved_params"]
    for key in set(manifest["defaults"]["dnn_defaults"]) & set(expected):
        if saved.get(key) != expected[key]:
            raise ValueError(f"Fold {fold}: saved {key}={saved.get(key)!r}, expected {expected[key]!r}")


def train(manifest, artifacts, status, scratch):
    models = Path(manifest["model_dir"])
    trainer = Path(manifest["source"]) / "hzgml/scripts" / manifest["training_script"]
    shared_metadata = models / "DNN_metadata_ggH.json"
    all_hashes = {}
    for fold in manifest["defaults"]["folds"]:
        receipt = status / f"fold_{fold}.json"
        names = [pattern.format(fold=fold) for pattern in FOLD_PRODUCTS]
        try:
            hashes = products(models, names) if receipt.is_file() else None
        except FileNotFoundError:
            hashes = None
        if hashes is not None and hashes == read(receipt):
            print(f"Reusing validated fold {fold}", flush=True)
        else:
            log_resources(f"before fold {fold}", scratch)
            with tempfile.TemporaryDirectory(prefix=f"pt100_fold_{fold}_", dir=scratch) as cache:
                command = [sys.executable, trainer, "-r", "ggH", "-c", manifest["config"],
                           "-o", models, "-f", fold, "--seed", manifest["defaults"]["seed"],
                           "--ptdecor", "--save", "--memory-mode", "streaming", "--cache-format", "npy",
                           "--no-corr", "--tensor-prefetch-workers", "4", "--tensor-prefetch-mb", "64",
                           "--train-cache-dir", cache]
                run(command, artifacts)
                log_resources(f"after fold {fold}", scratch)
            check_model(manifest, models, fold)
            hashes = products(models, names)
            write_json(receipt, hashes)
        all_hashes.update(hashes)
    shutil.copyfile(models / "DNN_metadata_ggH_3.json", shared_metadata)
    all_hashes[shared_metadata.name] = digest(shared_metadata)
    provenance = artifacts / "training_provenance.json"
    write_json(provenance, {"model_sha256": all_hashes, "resolved_params": manifest["resolved_params"]})
    return digest(provenance)


def worker(manifest_path):
    manifest = read(manifest_path)
    artifacts = Path(manifest["artifacts"])
    scratch = Path(os.environ["_CONDOR_SCRATCH_DIR"])
    status = artifacts / "receipts"
    status.mkdir(exist_ok=True)
    source = Path(manifest["source"])
    for relative, expected in manifest["sha256"].items():
        if digest(source / relative) != expected:
            raise ValueError(f"Source snapshot changed: {relative}")
    log_resources("startup", scratch)
    write_json(artifacts / "completion.json", {"completed": False, "stage": "training"})
    lineage = train(manifest, artifacts, status, scratch)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    write_json(artifacts / "completion.json", {"completed": False, "stage": "applying"})

    pilot_file = artifacts / "inference_pilot.json"
    pilot = read(pilot_file) if pilot_file.is_file() else {}
    if pilot.get("lineage") != lineage:
        checks = []
        for year in ("2022", "2024"):
            apply(manifest, year, pilot=True)
            checks.append(validate_apply(manifest, year, "GluGluHToMuMu_M125", pilot=True))
        if not all(sum(flow_counts[key] for check in checks for flow_counts in check["migrations"].values()) > 0
                   for key in ("in", "out", "both")):
            raise ValueError("Inference pilot needs both migration directions and stable events")
        write_json(pilot_file, {"passed": True, "lineage": lineage, "checks": checks})

    outputs = artifacts / "outputs"
    categories = manifest["categories"]
    files = [f"ggH/{category}.root{suffix}" for category in categories for suffix in ("", ".summary.json")]
    for year in manifest["years"]:
        directory = outputs / year
        receipt = status / f"apply_{year}.json"
        year_files = files + ["validation.json"]
        transformers = [f"../../transformers/{year}/DNN_tsf_ggH_{fold}.pkl" for fold in manifest["defaults"]["folds"]]
        if has_receipt(receipt, lineage, directory, year_files + transformers):
            print(f"Reusing validated Apply {year}", flush=True)
            continue
        log_resources(f"before Apply {year}", scratch)
        apply(manifest, year)
        checks = [validate_apply(manifest, year, name) for name in categories]
        write_json(directory / "validation.json", checks)
        save_receipt(receipt, lineage, directory, year_files + transformers)
        log_resources(f"after Apply {year}", scratch)

    write_json(artifacts / "completion.json", {"completed": False, "stage": "merging"})
    merged = ["report.json"] + [f"outputs/{category}.root{suffix}" for category in categories
                                for suffix in ("", ".summary.json")]
    receipt = status / "merge.json"
    if not has_receipt(receipt, lineage, artifacts, merged):
        merge(manifest, mark_complete=False)
        save_receipt(receipt, lineage, artifacts, merged)
    log_resources("after merge", scratch)
    if not manifest.get("defer_comparison"):
        run([sys.executable, source / "hzgml/scripts/pt100_grl_comparison.py", manifest_path], artifacts)
    write_json(artifacts / "completion.json", {
        "completed": True, "stage": "validated", "skim_files": len(manifest["records"]),
        "per_era_fit_files": len(manifest["records"]), "merged_files": len(categories),
        "comparison": "pending" if manifest.get("defer_comparison") else "grl_comparison.json",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("tag")
    job = commands.add_parser("worker")
    job.add_argument("manifest", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        configure(args.tag)
    else:
        worker(args.manifest)


if __name__ == "__main__":
    main()
