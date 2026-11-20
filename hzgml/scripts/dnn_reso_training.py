"""Resolution-weighted training policy and independent input audit."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import uproot

RESOLUTION = "diMu-mass_resolution_BSC"


def weight_policy(manifest):
    settings = manifest.get("training_weights", manifest.get("defaults", {}))
    return {"reso": bool(settings.get("reso", False)),
            "reso_bkg": bool(settings.get("reso_bkg", False)),
            "resolution_branch": RESOLUTION,
            "definition": "signed eventWeight after era scaling; signal divided by rounded resolution when reso; background divided only when reso_bkg; then existing train-reference class normalization",
            "invalid_resolution": "nonfinite or rounded resolution <= 0: effective training weight = 0",
            "fitting_eventWeight": "physical signed eventWeight after era scaling, without resolution division"}


def training_flags(manifest):
    policy = weight_policy(manifest)
    return [flag for key, flag in [("reso", "--reso"), ("reso_bkg", "--reso-bkg")] if policy[key]]


def resolution_method(script):
    module = ast.parse(Path(script).read_text())
    handler = next(x for x in module.body if isinstance(x, ast.ClassDef) and x.name == "DNNHandler")
    method = next(x for x in handler.body if isinstance(x, ast.FunctionDef) and x.name == "_apply_resolution_weight")
    namespace = {"np": np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(script), "exec"), namespace)
    return namespace[method.name]


def resolution_preflight(manifest):
    """Scan every selected signal event and verify the trainer's division independently."""
    config = runpy.run_path(manifest.get("training_config", manifest["config"]))["CONFIG"]["ggH"]
    policy = weight_policy(manifest)
    if not policy["reso"] and not policy["reso_bkg"]:
        return {"passed": True, "training_weights": policy, "samples": []}
    script = Path(manifest["source"]) / "hzgml/scripts" / manifest["training_script"]
    apply_resolution = resolution_method(script)
    weight = config.get("weight", "eventWeight")
    proxy = SimpleNamespace(reso=RESOLUTION, weight=weight)
    categories = set(config["train_signal"] if policy["reso"] else [])
    if policy["reso_bkg"]:
        categories.update(config.get("train_mc_background", []) + config.get("train_data_background", []))
    records = []
    for year in config["yearly_inputs"]:
        aliases = dict(config.get("branch_aliases", {}), **year.get("branch_aliases", {}))
        for category in sorted(categories):
            files = next(x["files"] for x in manifest.get("training_input_inventory", manifest["input_inventory"])
                         if x["year"] == year["year"] and x["category"] == category)
            record = {"year": year["year"], "category": category, "inputs": files,
                      "selected_entries": 0, "invalid_resolution": 0, "sum_physical_weight": 0.,
                      "sum_resolution_weight": 0., "negative_physical_weights": 0}
            for path in files:
                with uproot.open(path) as source:
                    tree = source[year.get("tree_map", {}).get(category, year.get("tree_name", "Events"))]
                    columns = [RESOLUTION, weight, "diMufsr_kit_BSC_mass", "cate_index"]
                    mapping = {aliases.get(name, name): name for name in columns}
                    missing = set(mapping) - set(tree.keys())
                    if missing:
                        raise KeyError(f"Resolution training preflight: {path} missing {sorted(missing)}")
                    for raw in tree.iterate(list(mapping), step_size="32 MB", library="np"):
                        data = pd.DataFrame({mapping[k]: v for k, v in raw.items()})
                        data = data.query(" and ".join(config["preselections"])).copy()
                        scale = float(year.get("eventWeight_scale", 1.)) if category not in config.get("train_data_background", []) else 1.
                        data[weight] = data[weight].astype(np.float64) * scale
                        physical = data[weight].to_numpy(copy=True)
                        reso = np.round(data[RESOLUTION].to_numpy(dtype=np.float64), 6)
                        valid = np.isfinite(reso) & (reso > 0)
                        expected = np.zeros_like(physical)
                        np.divide(physical, reso, out=expected, where=valid)
                        expected[~np.isfinite(expected)] = 0.
                        apply_resolution(proxy, data, f"preflight {year['year']}/{category}")
                        np.testing.assert_array_equal(data[weight].to_numpy(), expected)
                        record["selected_entries"] += len(data)
                        record["invalid_resolution"] += int(np.count_nonzero(~valid))
                        record["negative_physical_weights"] += int(np.count_nonzero(physical < 0))
                        record["sum_physical_weight"] += float(physical.sum(dtype=np.float64))
                        record["sum_resolution_weight"] += float(expected.sum(dtype=np.float64))
            records.append(record)
            print("RESO PREFLIGHT " + json.dumps(record), flush=True)
    return {"passed": True, "training_weights": policy, "samples": records,
            "trainer_sha256": hashlib.sha256(script.read_bytes()).hexdigest()}


def validate_model_policy(manifest):
    expected = weight_policy(manifest)
    models = []
    for fold in range(4):
        path = Path(manifest["model_dir"]) / f"DNN_metadata_ggH_{fold}.json"
        metadata = json.loads(path.read_text())
        for key in ["reso", "reso_bkg"]:
            if metadata.get("training_weights", {}).get(key) != expected[key]:
                raise ValueError(f"Training weight policy mismatch: {path}:{key}")
        model = Path(manifest["model_dir"]) / f"DNN_ggH_{fold}.keras"
        models.append({"fold": fold, "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                       "scaler_sha256": hashlib.sha256((model.parent / f"DNN_scaler_ggH_{fold}.pkl").read_bytes()).hexdigest()})
    return {"training_weights": expected, "models": models}
