"""Verify the frozen package, execute one DAG stage and checksum reusable artifacts."""

import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("stage")
    p.add_argument("--era")
    p.add_argument("--index", type=int)
    a = p.parse_args()
    run = a.run_dir
    snapshot = Path(__file__).resolve().parents[2]
    hash_path = run / "code_sha256.json" if snapshot.name == "code_snapshot" else snapshot.with_suffix(".sha256.json")
    package_hashes = read(hash_path)
    execution_tool_id = hashlib.sha256(json.dumps(package_hashes, sort_keys=True).encode()).hexdigest()
    for relative, digest in package_hashes.items():
        if sha(snapshot / relative) != digest:
            raise ValueError(f"Frozen code changed: {relative}")
    if a.stage == "await-scan":
        record = read(run / "inputs.json")["records"][a.index]
        receipt = run / "stage_checksums" / f"{a.era}_scan_{a.index}.json"
        provenance = read(run / "covariance_update.json")
        cluster = provenance["scan_jobs"][str(a.index)]
        began = time.monotonic()
        while True:
            queued = subprocess.check_output(["condor_q", str(cluster), "-json"], text=True)
            jobs = json.loads(queued) if queued.strip() else []
            if not jobs:
                break
            if any(j["JobStatus"] in (3, 5) for j in jobs):
                raise RuntimeError(f"Original scan {cluster} failed or held")
            if time.monotonic() - began > 20 * 3600:
                raise TimeoutError(f"Original scan {cluster} not finished after 20 hours")
            print("AWAIT_ORIGINAL_SCAN", cluster, flush=True)
            time.sleep(30)
        original = read(receipt)
        if original["identity"] != dict(tool_id=provenance["original_tool_id"], record=record, stage="scan"):
            raise ValueError("Original scan identity mismatch")
        for relative, digest in original["outputs"].items():
            if sha(run / relative) != digest:
                raise ValueError("Original scan checksum mismatch: " + relative)
        for item in record["inputs"]:
            st = Path(item["path"]).stat()
            if st.st_size != item["size"] or st.st_mtime_ns != item["mtime_ns"]:
                raise ValueError("Input changed: " + item["path"])
        ledger = run / "stage_checksums" / f"{a.era}_await-scan_{a.index}.json"
        ledger.write_text(json.dumps(dict(tool_id=execution_tool_id, source_receipt=str(receipt), source_sha256=sha(receipt)), indent=2) + "\n")
        # The old scheduler remains held until its original scans have left the queue.
        q = subprocess.check_output(["condor_q", "-constraint", f"DAGManJobId=={provenance['original_dag']}", "-json"], text=True)
        if not (json.loads(q) if q.strip() else []):
            subprocess.run(["condor_rm", str(provenance["original_dag"])], check=True)
        return
    paths = []
    record = None
    if a.stage in ("scan", "histogram", "fit-histogram", "ggf-histogram"):
        source = "inputs.json" if a.stage == "scan" else "validation_inputs.json"
        record = read(run / source)["records"][a.index]
        for item in record["inputs"]:
            stat = Path(item["path"]).stat()
            if stat.st_size != item["size"] or stat.st_mtime_ns != item["mtime_ns"]:
                raise ValueError(f"Input changed: {item['path']}")
        era, region, sample = record["era"], record["region"], record["sample"]
        if a.stage == "scan":
            base = run / "shared/cache" / era / region / (sample + "__all.npz")
        elif a.stage == "histogram":
            base = (
                run
                / era
                / "E/validation_candidate"
                / era
                / "raw"
                / region
                / (sample + ".npz")
            )
        elif a.stage == "fit-histogram":
            base = run / "fit_bin_plots/raw" / f"{a.index:03d}.npz"
        else:
            base = (
                run
                / era
                / "E/validation_candidate"
                / era
                / "ggf_dnn_inputs/raw"
                / region
                / (sample + ".npz")
            )
        paths = [base, base.with_suffix(".json")]
    identity = dict(
        tool_id=execution_tool_id, record=record, stage=a.stage
    )
    if paths and a.era and a.stage != "scan":
        identity["fit_sha256"] = sha(run / a.era / "E/fit/fit_results.json")
    ledger = (
        run
        / "stage_checksums"
        / f"{a.era or 'all'}_{a.stage}_{a.index if a.index is not None else 'all'}.json"
    )
    if paths and ledger.exists():
        saved = read(ledger)
        if saved["identity"] == identity and all(
            path.exists()
            and sha(path) == saved["outputs"].get(str(path.relative_to(run)))
            for path in paths
        ):
            print("VERIFIED_CACHE", ledger, flush=True)
            return
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).with_name("campaign.py")),
        "--run-dir",
        str(run),
        a.stage,
    ]
    if a.era:
        command += ["--era", a.era]
    if a.index is not None:
        command += ["--index", str(a.index)]
    os.environ["RECOIL_EXECUTION_TOOL_ID"] = execution_tool_id
    subprocess.run(command, check=True)
    if paths:
        outputs = {str(path.relative_to(run)): sha(path) for path in paths}
        ledger.parent.mkdir(exist_ok=True)
        temporary = ledger.with_suffix(".partial")
        temporary.write_text(
            json.dumps(dict(identity=identity, outputs=outputs), indent=2) + "\n"
        )
        temporary.replace(ledger)


if __name__ == "__main__":
    main()
