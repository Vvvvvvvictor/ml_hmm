#!/usr/bin/env python3
"""Recover a stopped campaign's fit inputs on shared storage without plotting."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def campaign_identity(run):
    hashes = read(run / "code_sha256.json")
    tool_id = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    if tool_id != read(run / "campaign.json")["tool_id"]:
        raise ValueError("Frozen campaign identity changed")
    for relative, digest in hashes.items():
        if sha(run / "code_snapshot" / relative) != digest:
            raise ValueError(f"Frozen code changed: {relative}")
    return tool_id


def scan_info(run, record, index, tool_id):
    era, region, sample = (record[key] for key in ("era", "region", "sample"))
    name = f"{era}_scan_{index}"
    receipt = run / "stage_checksums" / f"{name}.json"
    base = run / "shared" / "cache" / era / region / f"{sample}__all"
    paths = [base.with_name(base.name + suffix) for suffix in (".npz", ".json")]
    for item in record["inputs"]:
        stat = Path(item["path"]).stat()
        if stat.st_size != item["size"] or stat.st_mtime_ns != item["mtime_ns"]:
            raise ValueError(f"Frozen input changed: {item['path']}")
    return name, receipt, paths, dict(tool_id=tool_id, record=record, stage="scan")


def digests(run, paths):
    return {str(path.relative_to(run)): sha(path) for path in paths}


def verified(run, receipt, paths, identity):
    if not receipt.is_file():
        return False
    saved = read(receipt)
    return (saved["identity"] == identity and all(path.is_file() for path in paths)
            and saved["outputs"] == digests(run, paths))


def run_stage(run, control, stage, index, era):
    plan = read(control / "fit_inputs_plan.json")
    tool_id = campaign_identity(run)
    if tool_id != plan["tool_id"]:
        raise ValueError("Campaign identity differs from recovery plan")
    records = read(run / "inputs.json")["records"]
    program = run / "code_snapshot/scripts/Jet_RecoilCorrection/campaign.py"
    command = [sys.executable, "-B", str(program), "--run-dir", str(run)]
    if stage == "scan":
        name, receipt, paths, identity = scan_info(run, records[index], index, tool_id)
        if verified(run, receipt, paths, identity):
            print("VERIFIED_CACHE", name)
            return
        subprocess.run([*command, "scan", "--index", str(index)], check=True)
        write(receipt, dict(identity=identity, outputs=digests(run, paths)))
    elif stage == "preflight":
        subprocess.run([*command, "preflight", "--era", era], check=True)
        write(control / f"{era}_preflight_passed.json", dict(era=era, tool_id=tool_id))
    else:
        for label in plan["eras"]:
            if read(control / f"{label}_preflight_passed.json") != dict(era=label, tool_id=tool_id):
                raise ValueError(f"Missing current preflight: {label}")
        for i, record in enumerate(records):
            name, receipt, paths, identity = scan_info(run, record, i, tool_id)
            if not verified(run, receipt, paths, identity):
                raise ValueError(f"Fit input checksum mismatch: {name}")
        write(run / "fit_inputs_completion.json", dict(
            fit_inputs_ready=True, tool_id=tool_id, eras=plan["eras"],
            samples=len(records), control_dir=str(control),
        ))


def prepare(run, control):
    previous = read(run / "submission.json")
    cluster = int(re.search(r"cluster (\d+)", previous["output"])[1])
    if subprocess.check_output(
        ["condor_q", "-constraint", f"ClusterId=={cluster} || DAGManJobId=={cluster}",
         "-af", "ClusterId"], text=True,
    ).strip():
        raise RuntimeError(f"Original DAG {cluster} is still active")
    tool_id = campaign_identity(run)
    records = read(run / "inputs.json")["records"]
    eras = read(run / "campaign.json")["eras"]
    reused, pending = [], {}
    for i, record in enumerate(records):
        name, receipt, paths, identity = scan_info(run, record, i, tool_id)
        paths[0].parent.mkdir(parents=True, exist_ok=True)
        if verified(run, receipt, paths, identity):
            reused.append(name)
        else:
            pending[name] = (i, record["era"])
    (run / "stage_checksums").mkdir(exist_ok=True)

    control.mkdir()
    previous_control = Path(previous["control_dir"])
    shutil.copy2(previous_control / "run.sh", control / "run.sh")
    (control / "stage.sub").write_text(
        (previous_control / "stage.sub").read_text().replace(str(previous_control), str(control))
    )
    shutil.copy2(Path(__file__), control / "recover_fit_inputs.py")
    lines = [f"NODE_STATUS_FILE {control}/pipeline.status 30 ALWAYS-UPDATE"]

    def node(name, args, parents=()):
        lines.extend([
            f"JOB {name} {control}/stage.sub",
            f'VARS {name} name="{name}" args="{control}/recover_fit_inputs.py '
            f'{run} {control} {args}"',
            f"RETRY {name} 1",
        ])
        if parents:
            lines.append(f"PARENT {' '.join(parents)} CHILD {name}")

    for name, (index, _) in pending.items():
        node(name, f"scan --index {index}")
    for era in eras:
        node(f"{era}_preflight", f"preflight --era {era}",
             [name for name, (_, label) in pending.items() if label == era])
    node("fit_inputs_ready", "finish", [f"{era}_preflight" for era in eras])
    (control / "pipeline.dag").write_text("\n".join(lines) + "\n")
    write(control / "fit_inputs_plan.json", dict(
        tool_id=tool_id, original_dag=cluster, eras=eras, reused=reused, pending=pending,
        recovery_script_sha256=sha(control / "recover_fit_inputs.py"),
    ))
    print(f"Verified {len(reused)} cached scans. Prepared {len(pending)} scans and {len(eras)} preflights.", flush=True)
    subprocess.run(
        ["condor_submit_dag", "-no_submit", "-maxjobs", "1000", "pipeline.dag"],
        cwd=control, check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("control", type=Path)
    parser.add_argument("stage", choices=("prepare", "scan", "preflight", "finish"))
    parser.add_argument("--index", type=int)
    parser.add_argument("--era")
    args = parser.parse_args()
    run, control = args.run.resolve(), args.control.resolve()
    if args.stage == "prepare":
        prepare(run, control)
    else:
        run_stage(run, control, args.stage, args.index, args.era)


if __name__ == "__main__":
    main()
