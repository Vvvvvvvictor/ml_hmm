#!/usr/bin/env python3
"""Prepare, run and validate an isolated default-DNN HTCondor DAG."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
from types import SimpleNamespace
import zipfile

from dnn_root_io import open_published_root
from dnn_reso_training import weight_policy, training_flags, resolution_preflight, validate_model_policy
from dnn_fit_output import systematic_plan, mapping_record


SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[2]
CONFIG = "hzgml/data/training_config_BDT_ggH_multiyear_template.py"
INPUTS = "hzgml/data/inputs_config_DNN_ggH.json"
SOURCES = [CONFIG, INPUTS, "hzgml/scripts/root_tools/hadd", "hzgml/scripts/root_tools/xrdcp", "hzgml/scripts/root_tools/eos", "hzgml/data/apply_inputs_config_ggH.py",
           "hzgml/data/inputs_config_higgs_extra_ggH.json"] + ["hzgml/scripts/" + name for name in (
    "train_dnn_ggH.py", "train_dnn_ggH_true_mass_decor.py", "apply_dnn_ggH.py", "dnn_fit_output.py",
    "weighted_quantile_transformer.py", "setup_dnn_gpu.sh", "test_dnn_fit_output.py",
    "dnn_fit_campaign.py", "dnn_root_io.py", "merge_dnn_fit_eras.py", "dnn_reso_training.py",
    "test_dnn_reso_training.py", "promote_dnn_reso.py")]



def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def run(command, cwd):
    print("RUN", json.dumps([str(x) for x in command]), flush=True)
    subprocess.run([str(x) for x in command], cwd=cwd, check=True)


def resolve_parameters(script, defaults, overrides):
    """Evaluate the trainer's actual parameter merger without importing TensorFlow."""
    tree = ast.parse(Path(script).read_text())
    handler = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DNNHandler")
    functions = [n for n in handler.body if isinstance(n, ast.FunctionDef)
                 and n.name in {"_default_params", "_merge_dnn_params"}]
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), str(script), "exec"), namespace)
    proxy = SimpleNamespace(dnn_defaults=defaults)
    proxy._default_params = lambda: namespace["_default_params"](proxy)
    resolved = namespace["_merge_dnn_params"](proxy, overrides)
    for key, value in overrides.items():
        if key not in resolved or resolved[key] != value:
            raise ValueError(f"Trainer does not apply requested parameter {key}={value!r}")
    return resolved


def prepare(args):
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.run_dir or REPO / "condor/dnn/default_fit" / tag).resolve()
    artifacts = Path(args.artifacts or f"/eos/user/j/jiehan/HMuMuShare/DNNDefaultFit_Run3/{tag}").resolve()
    if run_dir.exists() or artifacts.exists():
        raise FileExistsError("Use new run and artifact directories; existing results are protected")
    config = runpy.run_path(str(REPO / CONFIG))["CONFIG"]["ggH"]
    training_script = args.training_script
    requested = None
    if args.hyperparams_file:
        requested = json.loads(Path(args.hyperparams_file).read_text())["best_params"]
        if not isinstance(requested, dict) or not requested:
            raise ValueError("Hyperparameter JSON must contain a non-empty best_params object")
    resolved = resolve_parameters(REPO / "hzgml/scripts" / training_script,
                                  config["dnn_defaults"], requested or {})
    categories = list(json.loads((REPO / INPUTS).read_text())["sample_list"])
    years = config["yearly_inputs"]
    inventory = []
    for year in years:
        for category in categories:
            names = [year.get("sample_map", {}).get(category, category), category]
            matches = sorted({str(p) for p in Path(year["folder"]).glob("*.root")
                              if any(p.name.endswith(name + ".root") for name in names)})
            if not matches:
                raise FileNotFoundError(f"No inputs for {year['year']}/{category}")
            inventory.append({"year": year["year"], "category": category, "files": matches})
    (run_dir / "logs").mkdir(parents=True)
    artifacts.mkdir(parents=True)
    snapshot = run_dir / "source"
    digests = {}
    for relative in SOURCES:
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, destination)
        digests[relative] = hashlib.sha256(destination.read_bytes()).hexdigest()
    hyperparams_path = None
    if requested is not None:
        relative = "hyperparams/DNN_region_ggH_all_folds.json"
        destination = snapshot / relative
        destination.parent.mkdir(parents=True)
        shutil.copy2(args.hyperparams_file, destination)
        digests[relative] = hashlib.sha256(destination.read_bytes()).hexdigest()
        hyperparams_path = str(destination.parent)
    manifest = {"created_utc": tag, "run_dir": str(run_dir), "artifacts": str(artifacts),
                "source": str(snapshot), "config": str(snapshot / CONFIG),
                "inputs_config": str(snapshot / INPUTS), "model_dir": str(artifacts / "models_dnn"),
                "categories": categories, "years": [y["year"] for y in years],
                "input_inventory": inventory, "sha256": digests,
                "training_script": training_script, "hyperparams_path": hyperparams_path,
                "hyperparams_source": str(Path(args.hyperparams_file).resolve()) if args.hyperparams_file else None,
                "requested_best_params": requested, "resolved_params": resolved,
                "training_weights": {"reso": args.reso, "reso_bkg": args.reso_bkg},
                "defaults": {"folds": [0, 1, 2, 3], "reso": args.reso, "reso_bkg": args.reso_bkg, "ptdecor": False,
                             "optuna": False, "seed": 20250408,
                             "dnn_defaults": config["dnn_defaults"]}}
    manifest_path = run_dir / "manifest.json"
    write_json(manifest_path, manifest)
    wrapper = run_dir / "worker.sh"
    # Paths are passed as arguments, never interpolated into shell program text.
    wrapper.write_text('#!/bin/bash\nset -eo pipefail\n'
                       'manifest="$1"\nstage="$2"\nyear="${3:-}"\nsource_root="$4"\n'
                       'source "${source_root}/hzgml/scripts/setup_dnn_gpu.sh"\n'
                       'export PYTHONUNBUFFERED=1\nexport PYTHONDONTWRITEBYTECODE=1\n'
                       'export MPLCONFIGDIR=/tmp/mpl-topwsf\nexport PYTHONPYCACHEPREFIX=/tmp/pycache-topwsf\n'
                       'if [ "$stage" = "smoke" ] || [ "$stage" = "apply_extra" ]; then export CUDA_VISIBLE_DEVICES=-1; fi\n'
                       'export OMP_NUM_THREADS=4\nexport TF_NUM_INTRAOP_THREADS=4\n'
                       'export TF_NUM_INTEROP_THREADS=2\nunset PYTHONHOME\n'
                       'exec python3 "${source_root}/hzgml/scripts/dnn_fit_campaign.py" '
                       'worker --manifest "$manifest" --stage "$stage" --year "$year"\n')
    wrapper.chmod(0o755)
    dag = []
    nodes = [("TRAIN", "train", "", True, 48 * 1024),
             ("SMOKE", "smoke", "", False, 16 * 1024)]
    nodes += [("APPLY_" + y, "apply", y, True, 16 * 1024) for y in manifest["years"]]
    nodes += [("REPORT", "report", "", False, 4 * 1024)]
    for name, stage, year, gpu, memory in nodes:
        submit = run_dir / (name + ".sub")
        submit.write_text(
            "universe = vanilla\nnotification = Never\n"
            f"initialdir = {run_dir}\nexecutable = {wrapper}\n"
            f'arguments = "{manifest_path} {stage} {year or "none"} {snapshot}"\n'
            f"output = {run_dir}/logs/{name}.$(ClusterId).out\n"
            f"error = {run_dir}/logs/{name}.$(ClusterId).err\n"
            f"log = {run_dir}/logs/{name}.$(ClusterId).log\n"
            "should_transfer_files = NO\nrequest_cpus = 4\n"
            f"request_memory = {memory}\nrequest_disk = 30000000\n"
            '+JobFlavour = "nextweek"\n'
            + ("request_gpus = 1\nrequirements = TARGET.GPUs_Capability >= 7.0\n" if gpu else "")
            + "queue\n")
        dag.append(f"JOB {name} {submit}")
    dag += ["PARENT TRAIN CHILD SMOKE",
            "PARENT SMOKE CHILD " + " ".join("APPLY_" + y for y in manifest["years"]),
            "PARENT " + " ".join("APPLY_" + y for y in manifest["years"]) + " CHILD REPORT"]
    dag_path = run_dir / "campaign.dag"
    dag_path.write_text("\n".join(dag) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "dag": str(dag_path), "artifacts": str(artifacts)}, indent=2))
    if args.submit:
        run(["condor_submit_dag", "-maxjobs", "7", str(dag_path)], run_dir)


def apply_command(manifest, year, output, *extra):
    config = runpy.run_path(manifest["config"])["CONFIG"]["ggH"]
    year_config = next(y for y in config["yearly_inputs"] if y["year"] == year)
    return [sys.executable, Path(manifest["source"]) / "hzgml/scripts/apply_dnn_ggH.py",
            "-c", manifest["config"], "--inputs-config", manifest["inputs_config"],
            "-i", year_config["folder"], "-m", manifest["model_dir"], "-o", output,
            "--year", year, "--exact-output-folder", *extra]


def apply_extra_command(manifest, year, output, *extra):
    config = runpy.run_path(manifest["extra_config"])["CONFIG"]["ggH"]
    year_config = next(y for y in config["yearly_inputs"] if y["year"] == year)
    return [sys.executable, Path(manifest["source"]) / "hzgml/scripts/apply_dnn_ggH.py",
            "-c", manifest["extra_config"], "--inputs-config", manifest["extra_inputs_config"],
            "-i", year_config["folder"], "-m", manifest["model_dir"], "-o", output,
            "--year", year, "--exact-output-folder", *extra]


def smoke(manifest):
    import numpy as np
    import uproot

    artifacts = Path(manifest["artifacts"])
    for year in ["2022", "2024"]:
        base = artifacts / "smoke" / year
        tsfs = base / "transformers"
        sample = "GluGluHToMuMu_M125"
        common = ["--max-chunks", "1", "--chunk-size", "512", "-cat", sample]
        run(apply_command(manifest, year, base / "fit", *common, "--systematics",
                          "--output-profile", "fit", "--save-score-transformers", str(tsfs)), artifacts)
        sidecar = json.loads((base / "fit" / "ggH" / (sample + ".root.summary.json")).read_text())
        with open_published_root(base / "fit" / "ggH" / (sample + ".root")) as fit_file:
            compact = fit_file["test"].arrays(library="np")
        # Check shifted features independently through the existing full-output inference path.
        from apply_dnn_ggH import ApplyDNNHandler
        from unittest.mock import patch
        import apply_dnn_ggH
        with patch.object(sys, "argv", [str(SCRIPT), *[str(x) for x in apply_command(
                manifest, year, base / "manual", *common, "--load-score-transformers", str(tsfs))[2:]]]):
            handler = ApplyDNNHandler(apply_dnn_ggH.getArgs())
        handler.loadModels()
        handler.loadScalers()
        handler.fitApplyScoreTransformers()
        with uproot.open(sidecar["inputs"][0]) as source:
            tree = source[handler._resolve_tree_name(source)]
            mapping = handler._resolve_branch_mapping(tree)
            branches = set(mapping)
            for variation in sidecar["systematics"].values():
                branches.add(variation["weight"])
                branches.update(variation["features"].values())
            raw = tree.arrays(sorted(branches), entry_stop=512, library="pd")
        # Legacy apply reads every input branch. Project out unrelated jagged SF arrays
        # so its original inference can be checked without an awkward-pandas dependency.
        scalar_input = base / "scalar_reference.root"
        arrays = {c: raw[c].to_numpy() for c in raw}
        with uproot.recreate(scalar_input) as root_file:
            tree = root_file.mktree("Events", {c: a.dtype for c, a in arrays.items()})
            tree.extend(arrays)
        full_output = base / "legacy_reference.root"
        handler._process_files([str(scalar_input)], str(full_output), sample)
        with open_published_root(full_output) as full_file:
            full = full_file["test"].arrays(library="np")
        for branch in ["run", "lumi", "event", "source_year", "diMufsr_kit_BSC_mass", "eventWeight"]:
            np.testing.assert_array_equal(compact[branch], full[branch])
        np.testing.assert_allclose(compact["score_t"], full["dnn_score_t"], rtol=2e-5, atol=2e-6)
        np.testing.assert_array_equal(compact["score_t"], compact["dnn_score_t"])
        scale = handler._event_weight_scale
        for variation in sidecar["systematics"].values():
            np.testing.assert_array_equal(compact[variation["weight_output"]], full[variation["weight"]] * scale)
            if not variation["features"]:
                np.testing.assert_array_equal(compact[variation["score_output"]], compact["score_t"])
        selected = handler.preselect(handler._inject_source_year(raw[list(mapping)].rename(columns=mapping)))
        for name, variation in sidecar["systematics"].items():
            if not variation["features"]:
                continue
            data = raw.loc[selected.index, list(mapping)].rename(columns=mapping).copy()
            for variable, branch in variation["features"].items():
                data[variable] = raw.loc[selected.index, branch]
            data = handler._inject_source_year(data)
            data = handler._apply_2223_jet_patch(data)
            data = handler._apply_ptdecor(data)
            fold = (data[handler.randomIndex] % 314159 % 4).to_numpy(dtype=int)
            reference = np.empty(len(data), dtype=np.float32)
            for i in range(4):
                # Exercise all-fold prediction as in the original apply, then select test events.
                values = handler._transform_features(data[handler.train_variables], i)
                scores, _ = handler._predict_score_domain(values, i)
                transformed = handler.m_tsfs[i].transform(scores.reshape(-1, 1)).reshape(-1)
                reference[fold == i] = transformed[fold == i]
            np.testing.assert_allclose(compact[variation["score_output"]], reference, rtol=2e-5, atol=2e-6)
    write_json(artifacts / "smoke" / "validation.json", {"passed": True, "years": ["2022", "2024"],
               "checks": ["legacy nominal", "all JER/JES", "all weights", "fixed selection", "fit reader schema"]})


def smoke_extra(manifest):
    import tempfile
    import numpy as np
    import uproot

    categories = manifest["supplemental_categories"]
    scratch = os.environ.get("_CONDOR_SCRATCH_DIR")
    if not scratch:
        raise RuntimeError("_CONDOR_SCRATCH_DIR is required for supplemental smoke")
    for year in ["2022", "2024"]:
        with tempfile.TemporaryDirectory(prefix="higgs_smoke_", dir=scratch) as temporary:
            output = Path(temporary) / "fit"
            transformers = Path(manifest["artifacts"]) / "transformers" / year
            run(apply_extra_command(manifest, year, output, "-cat", *categories,
                                    "--output-profile", "fit", "--systematics", "--max-chunks", "1",
                                    "--chunk-size", "512", "--load-score-transformers", transformers),
                manifest["artifacts"])
            for category in categories:
                root_path = output / "ggH" / f"{category}.root"
                summary = json.loads(Path(str(root_path) + ".summary.json").read_text())
                with uproot.open(root_path) as root_file:
                    tree = root_file["test"]
                    if tree.num_entries != summary["entries"] or len(summary["systematics"]) != 32:
                        raise ValueError(f"Supplemental smoke schema/count mismatch: {year}/{category}")
                    for arrays in tree.iterate(library="np", step_size="1 MB"):
                        if not np.array_equal(arrays["score_t"], arrays["dnn_score_t"]):
                            raise ValueError(f"Supplemental smoke score aliases differ: {year}/{category}")
                        for name, values in arrays.items():
                            if name.startswith(("score_t", "eventWeight")) and not np.isfinite(values).all():
                                raise ValueError(f"Supplemental smoke non-finite {year}/{category}:{name}")
    write_json(Path(manifest["artifacts"]) / "smoke" / "higgs_extra_validation.json",
               {"passed": True, "years": ["2022", "2024"], "categories": categories,
                "checks": ["real inputs", "four saved folds", "32 systematics",
                           "fixed selection", "finite scores and weights"]})


def report(manifest, require_complete=False):
    import numpy as np
    import uproot

    rows = []
    config = runpy.run_path(manifest["config"])["CONFIG"]["ggH"]
    for year in manifest["years"]:
        for category in manifest["categories"]:
            output = Path(manifest["artifacts"]) / "outputs" / year / "ggH" / (category + ".root")
            sidecar = Path(str(output) + ".summary.json")
            if not sidecar.exists():
                rows.append({"year": year, "category": category, "status": "pending"})
                continue
            record = json.loads(sidecar.read_text())
            with open_published_root(output) as source:
                output_bytes = source.file.source.num_bytes
                tree = source["test"]
                if tree.num_entries != record["entries"] or list(tree.keys()) != record["branches"]:
                    raise ValueError(f"Output schema/count mismatch: {output}")
                sums = {c: 0.0 for c in record["sumw"]}
                for arrays in tree.iterate(library="np", step_size="64 MB"):
                    np.testing.assert_array_equal(arrays["score_t"], arrays["dnn_score_t"])
                    for name, values in arrays.items():
                        if name.startswith(("score_t", "eventWeight")) and not np.isfinite(values).all():
                            raise ValueError(f"Non-finite output {output}:{name}")
                    for c in sums:
                        sums[c] += float(np.sum(arrays[c], dtype=np.float64))
                    for variation in record["systematics"].values():
                        if not variation["features"]:
                            np.testing.assert_array_equal(arrays[variation["score_output"]], arrays["score_t"])
                for column, value in sums.items():
                    if not np.isclose(value, record["sumw"][column], rtol=1e-10, atol=1e-8):
                        raise ValueError(f"Weight sum mismatch: {output}:{column}")
                # Independently check every selected source event and every full weight.
                year_config = next(y for y in config["yearly_inputs"] if y["year"] == year)
                aliases = year_config.get("branch_aliases", {})
                canonical = ["run", "lumi", "event", "diMufsr_kit_BSC_mass", "cate_index", "eventWeight"]
                source_names = {aliases.get(c, c): c for c in canonical}
                source_weights = {v["weight"]: v["weight_output"] for v in record["systematics"].values()}
                scale = float(year_config.get("eventWeight_scale", 1.0)) if category != "data" else 1.0
                offset = 0
                for path in record["inputs"]:
                    with uproot.open(path) as input_file:
                        input_tree = input_file[year_config.get("tree_name", "Events")]
                        expected_plan = (systematic_plan(input_tree.keys(), config["train_variables"], aliases,
                            config.get("weight", "eventWeight"), allow_nominal_only=category in config.get("train_mc_background", []))
                            if category != "data" else {})
                        if mapping_record(expected_plan) != record["systematics"]:
                            raise ValueError(f"Incomplete systematic schema: {output}")
                        branches = sorted(set(source_names) | set(source_weights))
                        for raw in input_tree.iterate(branches, library="np", step_size="64 MB"):
                            data = {target: raw[actual] for actual, target in source_names.items()}
                            selected = ((data["cate_index"] == 4) & (data["diMufsr_kit_BSC_mass"] >= 110)
                                        & (data["diMufsr_kit_BSC_mass"] <= 150))
                            count = int(np.sum(selected))
                            if not count:
                                continue
                            columns = ["run", "lumi", "event", "source_year", "diMufsr_kit_BSC_mass", "eventWeight"]
                            columns += list(source_weights.values())
                            actual = tree.arrays(columns, entry_start=offset, entry_stop=offset + count, library="np")
                            for column in ["run", "lumi", "event", "diMufsr_kit_BSC_mass"]:
                                np.testing.assert_array_equal(actual[column], data[column][selected])
                            np.testing.assert_array_equal(actual["source_year"], np.full(count, manifest["years"].index(year)))
                            np.testing.assert_array_equal(actual["eventWeight"], data["eventWeight"][selected].astype(np.float64) * scale)
                            for branch, target in source_weights.items():
                                np.testing.assert_array_equal(actual[target], raw[branch][selected].astype(np.float64) * scale)
                            offset += count
                if offset != tree.num_entries:
                    raise ValueError(f"Nominal selection count mismatch: {output}")
            rows.append({"year": year, "category": category, "status": "complete",
                         "entries": record["entries"], "sumw": record["sumw"],
                         "bytes": output_bytes, "output": str(output),
                         "systematic_count": len(record["systematics"]),
                         "systematics_status": record.get("systematics_status", "legacy")})
    result = {"model_dir": manifest["model_dir"], "artifacts": manifest["artifacts"],
              "complete": all(r["status"] == "complete" for r in rows), "samples": rows,
              "training_provenance": validate_model_policy(manifest) if training_flags(manifest) else {"training_weights": weight_policy(manifest)}}
    write_json(Path(manifest["artifacts"]) / "report.json", result)
    lines = ["year\tcategory\tstatus\tentries\tsumw\tbytes"]
    lines += [f"{r['year']}\t{r['category']}\t{r['status']}\t{r.get('entries', '')}\t"
              f"{r.get('sumw', {}).get('eventWeight', '')}\t{r.get('bytes', '')}" for r in rows]
    (Path(manifest["artifacts"]) / "report.tsv").write_text("\n".join(lines) + "\n")
    print(json.dumps({"complete": result["complete"], "completed": sum(r["status"] == "complete" for r in rows),
                      "total": len(rows), "report": str(Path(manifest["artifacts"]) / "report.json")}))
    if require_complete and not result["complete"]:
        raise RuntimeError("Some expected outputs are missing; see report.json")


def worker(args):
    manifest = json.loads(Path(args.manifest).read_text())
    for relative, digest in manifest["sha256"].items():
        if hashlib.sha256((Path(manifest["source"]) / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source snapshot changed: {relative}")
    artifacts = Path(manifest["artifacts"])
    root_tools = Path(manifest["source"]) / "hzgml/scripts/root_tools"
    if (root_tools / "hadd").is_file():
        os.environ["PATH"] = str(root_tools) + os.pathsep + os.environ["PATH"]
    if args.stage == "train":
        write_json(artifacts / "resolution_preflight.json", resolution_preflight(manifest))
        print("TRAINING WEIGHT POLICY " + json.dumps(weight_policy(manifest)), flush=True)
        script = Path(manifest["source"]) / "hzgml/scripts" / manifest.get("training_script", "train_dnn_ggH.py")
        command = [sys.executable, script, "-c", manifest.get("training_config", manifest["config"]), "-r", "ggH", "-o", "models_dnn", "--save"]
        if manifest.get("hyperparams_path"):
            command += ["--hyperparams_path", manifest["hyperparams_path"]]
        command += training_flags(manifest)
        if os.environ.get("DNN_TRAIN_CACHE_DIR"):
            command += ["--train-cache-dir", os.environ["DNN_TRAIN_CACHE_DIR"]]
        run(command, artifacts)
        for fold in range(4):
            for filename in [f"DNN_ggH_{fold}.keras", f"DNN_scaler_ggH_{fold}.pkl", f"DNN_metadata_ggH_{fold}.json"]:
                if not (Path(manifest["model_dir"]) / filename).is_file():
                    raise FileNotFoundError(filename)
            if manifest.get("requested_best_params"):
                with zipfile.ZipFile(Path(manifest["model_dir"]) / f"DNN_ggH_{fold}.keras") as model:
                    saved = json.loads(model.read("config.json"))["config"]["hparams"]
                for key, value in manifest["requested_best_params"].items():
                    if saved.get(key) != value:
                        raise ValueError(f"Saved fold {fold} hyperparameter mismatch: {key}")
        if training_flags(manifest):
            write_json(artifacts / "training_provenance.json", validate_model_policy(manifest))
    elif args.stage == "smoke":
        run([sys.executable, "-m", "unittest", "discover", "-s",
             Path(manifest["source"]) / "hzgml/scripts", "-p", "test_dnn_fit_output.py", "-v"], artifacts)
        smoke(manifest)
        if manifest.get("supplemental_categories") and manifest.get("run_supplemental_smoke"):
            smoke_extra(manifest)
    elif args.stage == "apply_all":
        worker(SimpleNamespace(manifest=args.manifest, stage="apply", year=args.year))
        worker(SimpleNamespace(manifest=args.manifest, stage="apply_extra", year=args.year))
    elif args.stage == "promote":
        from promote_dnn_reso import promote
        promote(args.manifest)
    elif args.stage == "apply":
        if args.year not in manifest["years"]:
            raise ValueError(f"Unknown year: {args.year}")
        if training_flags(manifest):
            validate_model_policy(manifest)
        output = artifacts / "outputs" / args.year
        run(apply_command(manifest, args.year, output, "--output-profile", "fit", "--systematics",
                          "--chunk-size", "100000", "--save-score-transformers",
                          str(artifacts / "transformers" / args.year)), artifacts)
        for category in manifest["categories"]:
            if not (output / "ggH" / (category + ".root.summary.json")).exists():
                raise RuntimeError(f"Missing output after apply: {args.year}/{category}")
    elif args.stage == "smoke_extra":
        smoke_extra(manifest)
    elif args.stage == "apply_extra":
        if args.year not in manifest["years"]:
            raise ValueError(f"Unknown year: {args.year}")
        output = artifacts / "outputs" / args.year
        run(apply_extra_command(manifest, args.year, output, "-cat", *manifest["supplemental_categories"],
                                "--output-profile", "fit", "--systematics", "--chunk-size", "100000",
                                "--load-score-transformers", artifacts / "transformers" / args.year), artifacts)
        for category in manifest["supplemental_categories"]:
            if not (output / "ggH" / (category + ".root.summary.json")).exists():
                raise RuntimeError(f"Missing supplemental output after apply: {args.year}/{category}")
    elif args.stage == "merge_eras":
        script = Path(manifest["source"]) / "hzgml/scripts/merge_dnn_fit_eras.py"
        run([sys.executable, script, "--manifest", args.manifest], artifacts)
    elif args.stage == "report":
        report(manifest, require_complete=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--run-dir")
    prep.add_argument("--artifacts")
    prep.add_argument("--submit", action="store_true")
    prep.add_argument("--reso", action="store_true")
    prep.add_argument("--reso-bkg", action="store_true")
    prep.add_argument("--hyperparams-file", help="JSON whose best_params are applied to all four folds")
    prep.add_argument("--training-script", choices=["train_dnn_ggH.py", "train_dnn_ggH_true_mass_decor.py"],
                      default="train_dnn_ggH.py")
    work = commands.add_parser("worker")
    work.add_argument("--manifest", required=True)
    work.add_argument("--stage", choices=["train", "smoke", "apply", "report", "smoke_extra",
                                          "apply_extra", "apply_all", "merge_eras", "promote"], required=True)
    work.add_argument("--year", default="")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        worker(args)


if __name__ == "__main__":
    main()
