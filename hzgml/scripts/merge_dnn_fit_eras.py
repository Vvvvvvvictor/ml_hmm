#!/usr/bin/env python3
"""Merge per-era compact DNN outputs by process and retire validated inputs."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import uproot

from dnn_root_io import open_published_root
from dnn_reso_training import training_flags, validate_model_policy
from dnn_fit_output import systematic_plan, mapping_record


BASE_COLUMNS = ["run", "lumi", "event", "source_year", "diMufsr_kit_BSC_mass",
                "score_t", "eventWeight", "dnn_score_t"]
YEAR_MARKERS = ["2022ee", "2023bpix", "2022", "2023", "2024", "2025", "2026"]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def eos_command(*args, check=True):
    executable = shutil.which("eos")
    if not executable:
        raise RuntimeError("eos command is required for publishing and cleanup")
    result = subprocess.run([executable, "root://eosuser.cern.ch", *map(str, args)],
                            capture_output=True, text=True)
    if check and result.returncode:
        raise RuntimeError(f"eos {' '.join(map(str, args))} failed ({result.returncode})\n"
                           f"{result.stdout}\n{result.stderr}")
    return result


def publish(local_path, eos_path):
    eos_path = str(eos_path)
    eos_command("mkdir", "-p", str(Path(eos_path).parent))
    temporary = eos_path + f".tmp.{os.getpid()}"
    eos_command("rm", temporary, check=False)
    result = subprocess.run([shutil.which("xrdcp") or "xrdcp", "-f", str(local_path),
                             "root://eosuser.cern.ch/" + temporary], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"xrdcp publish failed for {eos_path}\n{result.stdout}\n{result.stderr}")
    try:
        eos_command("rm", eos_path, check=False)
        eos_command("mv", temporary, eos_path)
    except Exception:
        eos_command("rm", temporary, check=False)
        raise


def source_paths(artifacts, years, category):
    return [artifacts / "outputs" / year / "ggH" / f"{category}.root" for year in years]


def source_summaries(paths):
    summaries = []
    for path in paths:
        summary_path = Path(str(path) + ".summary.json")
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summaries.append(read_json(summary_path))
    return summaries


def systematic_columns(summaries, years, category):
    by_year = {}
    for year, summary in zip(years, summaries):
        mapping = {}
        for item in summary.get("systematics", {}).values():
            suffix = item["suffix"]
            mapping[suffix] = (item["score_output"], item["weight_output"])
        by_year[year] = mapping
    all_suffixes = set().union(*(set(x) for x in by_year.values()))
    if category in {"data"} or not all_suffixes:
        if all_suffixes:
            raise ValueError(f"Unexpected systematics for {category}")
        return {}, by_year

    common = set.intersection(*(set(x) for x in by_year.values()))
    for suffix in all_suffixes - common:
        marker = next((marker for marker in YEAR_MARKERS if marker in suffix), None)
        if not suffix.startswith("jes") or marker is None:
            raise ValueError(f"Only era-specific JES branches may be absent: {category}:{suffix}")
        owner = {"2022ee": "2022EE", "2023bpix": "2023BPix", "2022": "2022",
                 "2023": "2023", "2024": "2024", "2025": "2025", "2026": "2026"}[marker]
        if owner not in years or suffix not in by_year[owner]:
            raise ValueError(f"Could not identify owner era for {category}:{suffix}")
        for year, mapping in by_year.items():
            if year != owner and suffix in mapping:
                raise ValueError(f"Era-specific systematic appears outside its era: {year}:{suffix}")

    for year, mapping in by_year.items():
        for suffix, pair in mapping.items():
            if pair != ("score_t_" + suffix, "eventWeight_" + suffix):
                raise ValueError(f"Unexpected output branch mapping: {category}/{year}/{suffix}: {pair}")
    return {suffix: ("score_t_" + suffix, "eventWeight_" + suffix)
            for suffix in sorted(all_suffixes)}, by_year


def branch_types_for(paths, summaries, years, category, syst_columns):
    types = {}
    source_types = {}
    for year, path, summary in zip(years, paths, summaries):
        with open_published_root(path) as root_file:
            tree = root_file["test"]
            if list(tree.keys()) != summary["branches"]:
                raise ValueError(f"Source branch schema changed for {path}")
            for name in BASE_COLUMNS:
                if name not in tree:
                    raise ValueError(f"Missing required branch {name} in {path}")
            source_types[year] = {name: np.dtype(branch.interpretation.numpy_dtype)
                                  for name, branch in tree.items()}
            for name in BASE_COLUMNS:
                dtype = source_types[year][name]
                if name in types and np.dtype(types[name]) != np.dtype(dtype):
                    raise TypeError(f"Branch type differs for {category}:{name}")
                types[name] = np.dtype(dtype)
    for suffix, (score_name, weight_name) in syst_columns.items():
        score_types = []
        weight_types = []
        for year, summary in zip(years, summaries):
            available = suffix in {item["suffix"] for item in summary.get("systematics", {}).values()}
            score_branch = score_name if available else "score_t"
            weight_branch = weight_name if available else "eventWeight"
            if score_branch not in source_types[year] or weight_branch not in source_types[year]:
                raise ValueError(f"Missing systematic or nominal branch in {year}/{category}: {suffix}")
            score_types.append(source_types[year][score_branch])
            weight_types.append(source_types[year][weight_branch])
        if len(set(score_types)) != 1 or len(set(weight_types)) != 1:
            raise TypeError(f"Systematic branch types differ for {category}:{suffix}")
        types[score_name], types[weight_name] = score_types[0], weight_types[0]
    return types


def merge_one(artifacts, years, category, scratch, provenance=None):
    paths = source_paths(artifacts, years, category)
    summaries = source_summaries(paths)
    for year, summary in zip(years, summaries):
        if summary.get("year") != year or summary.get("category") != category:
            raise ValueError(f"Summary identity mismatch for {year}/{category}")
    syst_columns, by_year = systematic_columns(summaries, years, category)
    if category in {"GluGluHToMuMu_M125", "VBFHToMuMu_M125", "ttHToMuMu_M125",
                    "WplusHToMuMu_M125", "WminusHToMuMu_M125", "ZHToMuMu_M125"}:
        if len(syst_columns) != 92:
            raise ValueError(f"Expected 92 merged signal variations for {category}, got {len(syst_columns)}")
    branch_types = branch_types_for(paths, summaries, years, category, syst_columns)
    ordered = BASE_COLUMNS + [name for pair in syst_columns.values() for name in pair]
    if set(ordered) != set(branch_types) or len(ordered) != len(branch_types):
        raise ValueError(f"Merged branch schema mismatch for {category}")
    local = Path(scratch) / f"{category}.root"
    era_records = []
    sums = {name: 0.0 for name in ordered if name == "eventWeight" or name.startswith("eventWeight_")}
    branch_hashes = {name: hashlib.sha256() for name in ordered}
    entry_offset = 0
    with uproot.recreate(local) as destination:
        out_tree = destination.mktree("test", {name: branch_types[name] for name in ordered})
        for era_index, (year, path, summary) in enumerate(zip(years, paths, summaries)):
            era_start = entry_offset
            source_sumw = summary.get("sumw", {})
            era_sums = {name: 0.0 for name in source_sumw}
            expected_count = int(summary["entries"])
            with open_published_root(path) as root_file:
                tree = root_file["test"]
                if tree.num_entries != expected_count:
                    raise ValueError(f"Source entry count changed for {path}")
                available = {item["suffix"] for item in summary.get("systematics", {}).values()}
                for arrays in tree.iterate(list(tree.keys()), step_size="64 MB", library="np"):
                    output = {name: arrays[name] for name in BASE_COLUMNS}
                    if not np.all(arrays["source_year"] == era_index):
                        raise ValueError(f"source_year mismatch in {path}")
                    if not np.array_equal(arrays["score_t"], arrays["dnn_score_t"]):
                        raise ValueError(f"score aliases differ in {path}")
                    if not np.isfinite(arrays["diMufsr_kit_BSC_mass"]).all():
                        raise ValueError(f"Non-finite dimuon mass in {path}")
                    for name in arrays:
                        if name.startswith(("score_t", "eventWeight")) and not np.isfinite(arrays[name]).all():
                            raise ValueError(f"Non-finite branch {name} in {path}")
                    for suffix, (score_name, weight_name) in syst_columns.items():
                        if suffix in available:
                            output[score_name] = arrays[score_name]
                            output[weight_name] = arrays[weight_name]
                        else:
                            output[score_name] = arrays["score_t"]
                            output[weight_name] = arrays["eventWeight"]
                    output = {name: output[name] for name in ordered}
                    for name, values in output.items():
                        if name.startswith(("score_t", "eventWeight")) and not np.isfinite(values).all():
                            raise ValueError(f"Non-finite merged branch {name} in {path}")
                    out_tree.extend(output)
                    for name, values in output.items():
                        branch_hashes[name].update(np.ascontiguousarray(values).view(np.uint8))
                    for name in sums:
                        sums[name] += float(np.sum(output[name], dtype=np.float64))
                    for name in era_sums:
                        era_sums[name] += float(np.sum(output[name], dtype=np.float64))
                    entry_offset += len(arrays["event"])
            if entry_offset - era_start != expected_count:
                raise ValueError(f"Merged era entry count mismatch for {year}/{category}")
            for name, expected in source_sumw.items():
                if name not in era_sums or not np.isclose(era_sums[name], expected, rtol=1e-10, atol=1e-7):
                    raise ValueError(f"Source weight sum mismatch for {year}/{category}:{name}")
            era_records.append({"year": year, "source": str(path), "entries": expected_count,
                                "entry_start": era_start, "entry_stop": entry_offset,
                                "sumw": era_sums,
                                "nominal_filled_variations": sorted(set(syst_columns) - available)})

    final_path = artifacts / "outputs" / f"{category}.root"
    publish(local, final_path)
    with open_published_root(final_path) as root_file:
        tree = root_file["test"]
        if tree.num_entries != entry_offset or list(tree.keys()) != ordered:
            raise ValueError(f"Published merged ROOT failed schema/count validation: {final_path}")
        check_sums = {name: 0.0 for name in sums}
        readback_hashes = {name: hashlib.sha256() for name in ordered}
        for arrays in tree.iterate(list(tree.keys()), step_size="64 MB", library="np"):
            if not np.array_equal(arrays["score_t"], arrays["dnn_score_t"]):
                raise ValueError(f"Published score aliases differ in {final_path}")
            if not np.isfinite(arrays["diMufsr_kit_BSC_mass"]).all():
                raise ValueError(f"Published non-finite dimuon mass in {final_path}")
            for name in arrays:
                if name.startswith(("score_t", "eventWeight")) and not np.isfinite(arrays[name]).all():
                    raise ValueError(f"Published non-finite branch {name} in {final_path}")
            for name in check_sums:
                check_sums[name] += float(np.sum(arrays[name], dtype=np.float64))
            for name in ordered:
                readback_hashes[name].update(np.ascontiguousarray(arrays[name]).view(np.uint8))
        for name in ordered:
            if readback_hashes[name].digest() != branch_hashes[name].digest():
                raise ValueError(f"Published branch content differs from merged source: {final_path}:{name}")
        for name, value in sums.items():
            if not np.isclose(check_sums[name], value, rtol=1e-10, atol=1e-7):
                raise ValueError(f"Published sum mismatch {final_path}:{name}")
        size = root_file.file.source.num_bytes
    record = {"category": category, "status": "complete", "output": str(final_path),
              "tree": "test", "entries": entry_offset, "sumw": check_sums,
              "bytes": size, "branches": ordered, "systematic_count": len(syst_columns),
              "systematics": sorted(syst_columns), "eras": era_records,
              "missing_era_variations": "filled from nominal score_t and eventWeight",
              "full_branch_readback": True, "training_provenance": provenance}
    publish_summary = Path(scratch) / f"{category}.root.summary.json"
    publish_summary.write_text(json.dumps(record, indent=2) + "\n")
    publish(publish_summary, str(final_path) + ".summary.json")
    return record


def validate_source_selection(manifest, year, category):
    """Compare a newly applied sample against every nominally selected input event."""
    import runpy

    artifacts = Path(manifest["artifacts"])
    summary_path = artifacts / "outputs" / year / "ggH" / f"{category}.root.summary.json"
    record = read_json(summary_path)
    config_path = manifest["extra_config"] if category in manifest["supplemental_categories"] else manifest["config"]
    config = runpy.run_path(config_path)["CONFIG"]["ggH"]
    year_config = next(y for y in config["yearly_inputs"] if y["year"] == year)
    aliases = year_config.get("branch_aliases", {})
    canonical = ["run", "lumi", "event", "diMufsr_kit_BSC_mass", "cate_index", "eventWeight"]
    source_names = {aliases.get(name, name): name for name in canonical}
    source_weights = {item["weight"]: item["weight_output"]
                      for item in record.get("systematics", {}).values()}
    scale = float(year_config.get("eventWeight_scale", 1.0))
    output_path = artifacts / "outputs" / year / "ggH" / f"{category}.root"
    offset = 0
    with open_published_root(output_path) as output_file:
        output_tree = output_file["test"]
        if output_tree.num_entries != int(record["entries"]):
            raise ValueError(f"Output entry count differs from sidecar: {output_path}")
        for input_path in record["inputs"]:
            with uproot.open(input_path) as input_file:
                input_tree = input_file[year_config.get("tree_name", "Events")]
                expected = systematic_plan(input_tree.keys(), config["train_variables"], aliases,
                                           config.get("weight", "eventWeight"))
                if mapping_record(expected) != record["systematics"]:
                    raise ValueError(f"Input systematic mapping differs from apply output: {input_path}")
                branches = sorted(set(source_names) | set(source_weights))
                category_branch = aliases.get("cate_index", "cate_index")
                mass_branch = aliases.get("diMufsr_kit_BSC_mass", "diMufsr_kit_BSC_mass")
                for raw in input_tree.iterate(branches, library="np", step_size="64 MB"):
                    selected = ((raw[category_branch] == 4) & (raw[mass_branch] >= 110)
                                & (raw[mass_branch] <= 150))
                    count = int(np.sum(selected))
                    if not count:
                        continue
                    columns = ["run", "lumi", "event", "source_year", "diMufsr_kit_BSC_mass", "eventWeight"]
                    columns.extend(source_weights.values())
                    actual = output_tree.arrays(columns, entry_start=offset, entry_stop=offset + count,
                                                library="np")
                    for output_name, source_name in (("run", aliases.get("run", "run")),
                                                     ("lumi", aliases.get("lumi", "lumi")),
                                                     ("event", aliases.get("event", "event")),
                                                     ("diMufsr_kit_BSC_mass", aliases.get("diMufsr_kit_BSC_mass", "diMufsr_kit_BSC_mass"))):
                        np.testing.assert_array_equal(actual[output_name], raw[source_name][selected])
                    np.testing.assert_array_equal(actual["source_year"],
                                                  np.full(count, manifest["years"].index(year)))
                    weight_input = aliases.get("eventWeight", "eventWeight")
                    np.testing.assert_array_equal(actual["eventWeight"], raw[weight_input][selected] * scale)
                    for input_branch, output_branch in source_weights.items():
                        np.testing.assert_array_equal(actual[output_branch], raw[input_branch][selected] * scale)
                    offset += count
    if offset != int(record["entries"]):
        raise ValueError(f"Input nominal selection count {offset} differs from output {output_path}:{record['entries']}")
    return {"year": year, "category": category, "entries": offset, "status": "passed"}


def remove_per_era_outputs(artifacts, years, categories, audit_dir):
    state_path = audit_dir / "merge_cleanup.json"
    if state_path.exists():
        state = read_json(state_path)
    else:
        items = []
        for year in years:
            for category in categories:
                base = artifacts / "outputs" / year / "ggH" / f"{category}.root"
                items.extend([str(base), str(base) + ".summary.json"])
        state = {"complete_outputs_validated": True, "items": [{"path": p, "removed": False} for p in items]}
        write_json(state_path, state)
    if not state.get("complete_outputs_validated"):
        raise ValueError("Refusing per-era cleanup before merged outputs pass validation")
    for item in state["items"]:
        if item["removed"]:
            continue
        status = eos_command("stat", item["path"], check=False)
        if status.returncode:
            missing = ("no such file" in (status.stdout + status.stderr).lower()
                       or "not found" in (status.stdout + status.stderr).lower())
            if not missing:
                raise RuntimeError(f"Could not verify cleanup target {item['path']}\n"
                                   f"{status.stdout}\n{status.stderr}")
        else:
            eos_command("rm", item["path"])
        item["removed"] = True
        write_json(state_path, state)
    for year in years:
        path = artifacts / "outputs" / year / "ggH"
        eos_command("rmdir", str(path), check=False)
        eos_command("rmdir", str(path.parent), check=False)
    state["complete"] = True
    write_json(state_path, state)


def run(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    artifacts = Path(manifest["artifacts"])
    provenance = validate_model_policy(manifest) if training_flags(manifest) else None
    years = list(manifest["years"])
    categories = list(manifest["categories"]) + list(manifest["supplemental_categories"])
    audit_dir = Path(manifest["run_dir"]) / "merge_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    cleanup_state = audit_dir / "merge_cleanup.json"
    if cleanup_state.exists() and read_json(cleanup_state).get("complete_outputs_validated"):
        remove_per_era_outputs(artifacts, years, categories, audit_dir)
        report_path = artifacts / "report.json"
        report = read_json(report_path)
        report["cleanup"] = "complete"
        write_json(report_path, report)
        return

    old_report = artifacts / "report.json"
    old_tsv = artifacts / "report.tsv"
    if old_report.exists() and not (artifacts / "report.per_era.json").exists():
        write_json(artifacts / "report.per_era.json", read_json(old_report))
    if old_tsv.exists() and not (artifacts / "report.per_era.tsv").exists():
        (artifacts / "report.per_era.tsv").write_text(old_tsv.read_text())

    scratch = os.environ.get("_CONDOR_SCRATCH_DIR")
    if not scratch:
        raise RuntimeError("_CONDOR_SCRATCH_DIR is required for merge staging")
    validation_path = audit_dir / "supplemental_input_validation.json"
    if validation_path.exists():
        validation = read_json(validation_path)
        if not validation.get("passed") or len(validation.get("samples", [])) != 28:
            raise ValueError("Existing supplemental input validation record is incomplete")
    else:
        input_checks = []
        for category in manifest["supplemental_categories"]:
            for year in years:
                input_checks.append(validate_source_selection(manifest, year, category))
        validation = {"passed": len(input_checks) == 28 and all(x["status"] == "passed" for x in input_checks),
                      "samples": input_checks}
        write_json(validation_path, validation)
    if not validation.get("passed"):
        raise RuntimeError("Supplemental input selection validation failed")

    records = []
    for category in categories:
        existing_summary = artifacts / "outputs" / f"{category}.root.summary.json"
        existing_root = artifacts / "outputs" / f"{category}.root"
        if existing_summary.is_file() and existing_root.is_file():
            existing_record = read_json(existing_summary)
            if (existing_record.get("status") == "complete" and existing_record.get("full_branch_readback")
                    and existing_record.get("training_provenance") == provenance):
                records.append(existing_record)
                print(f"RESUMED {category} entries={existing_record['entries']} bytes={existing_record['bytes']}",
                      flush=True)
                continue
        with tempfile.TemporaryDirectory(prefix="dnn_era_merge_", dir=scratch) as work:
            records.append(merge_one(artifacts, years, category, work, provenance))
        print(f"MERGED {category} entries={records[-1]['entries']} bytes={records[-1]['bytes']}", flush=True)
    report = {"complete": len(records) == 22 and all(x["status"] == "complete" for x in records),
              "layout": "one test TTree per process, all eras concatenated",
              "years": years, "categories": categories, "samples": records,
              "supplemental_input_validation": str(audit_dir / "supplemental_input_validation.json"),
              "source_report": str(artifacts / "report.per_era.json"),
              "cleanup": "pending", "training_provenance": provenance}
    if not report["complete"]:
        raise RuntimeError("Merged report is incomplete")
    write_json(artifacts / "report.json", report)
    lines = ["category\tstatus\tentries\tsumw\tbytes\tsystematic_count"]
    lines.extend(f"{x['category']}\t{x['status']}\t{x['entries']}\t{x['sumw'].get('eventWeight', 0)}\t"
                 f"{x['bytes']}\t{x['systematic_count']}" for x in records)
    (artifacts / "report.tsv").write_text("\n".join(lines) + "\n")
    write_json(cleanup_state, {"complete_outputs_validated": True,
                               "items": [{"path": p, "removed": False} for year in years
                                         for category in categories
                                         for p in [str(artifacts / "outputs" / year / "ggH" / f"{category}.root"),
                                                   str(artifacts / "outputs" / year / "ggH" / f"{category}.root.summary.json") ] ]})
    remove_per_era_outputs(artifacts, years, categories, audit_dir)
    report["cleanup"] = "complete"
    write_json(artifacts / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    run(args.manifest)


if __name__ == "__main__":
    main()
