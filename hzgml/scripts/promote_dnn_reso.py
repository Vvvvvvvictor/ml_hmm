"""Publish a fully validated reso run while retaining the previous version."""
import json
from pathlib import Path
import subprocess

from dnn_reso_training import validate_model_policy


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def eos(*args, check=True):
    result = subprocess.run(["eos", "root://eosuser.cern.ch", *map(str, args)],
                            capture_output=True, text=True)
    if check and result.returncode:
        raise RuntimeError(f"EOS operation failed {args}: {result.stdout}\n{result.stderr}")
    return result


def exists(path):
    result = eos("stat", path, check=False)
    if result.returncode and not any(t in (result.stdout + result.stderr).lower()
                                     for t in ["no such file", "not found"]):
        raise RuntimeError(f"Cannot check {path}: {result.stdout}\n{result.stderr}")
    return result.returncode == 0


def relocate(value, source, target):
    if isinstance(value, str):
        return target + value[len(source):] if value == source or value.startswith(source + "/") else value
    if isinstance(value, list):
        return [relocate(v, source, target) for v in value]
    if isinstance(value, dict):
        return {k: relocate(v, source, target) for k, v in value.items()}
    return value


def promote(manifest_path):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    target = Path(manifest["promotion_target"])
    journal_path = Path(manifest["run_dir"]) / "promotion.json"
    if journal_path.exists():
        journal = json.loads(journal_path.read_text())
        if journal["status"] == "complete":
            return
    else:
        stage = Path(manifest["artifacts"])
        report = json.loads((stage / "report.json").read_text())
        if not report["complete"] or report.get("cleanup") != "complete" or len(report["samples"]) != 22:
            raise ValueError("Promotion requires 22 fully validated outputs and completed staging cleanup")
        validate_model_policy(manifest)
        for sample in report["samples"]:
            if not sample.get("full_branch_readback"):
                raise ValueError("Missing complete readback verification")
        names = ["models_dnn", "transformers", "outputs", "plots_dnn", "smoke", "smoke_extra",
                 "report.json", "report.tsv", "report.per_era.json", "report.per_era.tsv",
                 "training_provenance.json", "resolution_preflight.json"]
        backup = target / "backups" / ("pre_reso_" + manifest["created_utc"])
        eos("mkdir", "-p", backup)
        journal = {"status": "prepared", "staging": str(stage), "target": str(target),
                   "backup": str(backup), "archive": [n for n in names if exists(target / n)],
                   "publish": [n for n in names if exists(stage / n)], "moves": []}
        # Rewrite only JSON path metadata; ROOT contents have already passed readback validation.
        metadata = list(stage.glob("*.json")) + list((stage / "outputs").glob("*.summary.json"))
        for item in metadata:
            save(item, relocate(json.loads(item.read_text()), str(stage), str(target)))
        save(journal_path, journal)
    stage, backup = Path(journal["staging"]), Path(journal["backup"])
    journal["status"] = "publishing"
    save(journal_path, journal)

    def move(source, destination):
        operation = {"source": str(source), "destination": str(destination)}
        if operation in journal["moves"]:
            return
        if exists(source):
            if exists(destination):
                raise FileExistsError(destination)
            eos("mv", source, destination)
        elif not exists(destination):
            raise FileNotFoundError(source)
        journal["moves"].append(operation)
        save(journal_path, journal)

    try:
        for name in journal["archive"]:
            move(target / name, backup / name)
        for name in journal["publish"]:
            move(stage / name, target / name)
        delivered = json.loads((target / "report.json").read_text())
        if not delivered["complete"] or delivered.get("cleanup") != "complete":
            raise ValueError("Delivered report is incomplete")
        if len(list((target / "outputs").glob("*.root"))) != 22:
            raise ValueError("Delivered output inventory differs from 22 files")
    except Exception:
        errors = []
        for name in reversed(journal["publish"]):
            if not exists(stage / name) and exists(target / name):
                result = eos("mv", target / name, stage / name, check=False)
                if result.returncode:
                    errors.append(result.stderr)
        for name in journal["archive"]:
            if exists(backup / name) and not exists(target / name):
                result = eos("mv", backup / name, target / name, check=False)
                if result.returncode:
                    errors.append(result.stderr)
        journal["status"] = "rollback_failed" if errors else "rolled_back"
        journal["rollback_errors"] = errors
        journal["moves"] = []
        save(journal_path, journal)
        raise
    manifest["staging_artifacts"] = str(stage)
    manifest["artifacts"] = str(target)
    manifest["model_dir"] = str(target / "models_dnn")
    save(path, manifest)
    if manifest.get("previous_manifest"):
        previous_path = Path(manifest["previous_manifest"])
        previous = json.loads(previous_path.read_text())
        previous["active_reso_manifest"] = str(path)
        previous["published_training_weights"] = manifest["training_weights"]
        previous["resolution_retrain_status"] = "complete"
        save(previous_path, previous)
    save(target / "resolution_retrain.json", {"complete": True, "manifest": str(path),
                                            "backup": str(backup), "promotion": str(journal_path),
                                            "training_provenance": delivered["training_provenance"]})
    journal["status"] = "complete"
    save(journal_path, journal)
