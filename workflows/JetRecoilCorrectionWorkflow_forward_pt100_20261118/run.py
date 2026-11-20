#!/usr/bin/env python3
"""Freeze, prepare and submit the complete seven-era recoil campaign."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

CORE = Path(__file__).resolve().parent
PYTHON = "/eos/user/j/jiehan/ml-gpu-env/bin/python3"
PROXY = "/afs/cern.ch/user/j/jiehan/x509up_u152815"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def figure_catalog(eras):
    variables = read(CORE / "variables.json")
    figures = []
    for era in eras:
        figures.extend(
            p.replace("{era}", era) for p in read(CORE / "plots.json")["figures"]
        )
        for region, names in variables["regions"].items():
            base = f"{era}/E/validation_candidate/{era}/{region}"
            figures += [
                base + "/agreement_summary.pdf",
                base + "/match_split_summary.pdf",
            ]
            for name in names:
                name = "mu_met_pt" if name == "recoil" else name
                figures += [
                    f"{base}/{name}_{norm}.{ext}"
                    for norm in variables["normalizations"]
                    for ext in ("png", "pdf")
                ]
            for name in ("jet_pt", "jet_eta"):
                for stage in ("nominal", "corrected"):
                    for norm in variables["normalizations"]:
                        suffix = "" if norm == "absolute" else "_" + norm
                        figures += [
                            f"{base}/match_split/{name}_{stage}{suffix}.{ext}"
                            for ext in ("png", "pdf")
                        ]
            for kind in ("all", "leading", "subleading"):
                for name in ("jet_pt", "jet_eta"):
                    for component in ("matched", "unmatched", "fraction"):
                        figures += [f"{base}/match_split/{kind}_{name}_{component}_jet_sf_physical.{ext}" for ext in ("png", "pdf")]
        for region in ("zcr", "srsb"):
            base = f"{era}/E/validation_candidate/{era}/ggf_dnn_inputs/{region}"
            figures.append(base + "/summary.pdf")
            figures += [
                f"{base}/{name}_{norm}.{ext}"
                for name in variables["ggf"]
                for norm in variables["normalizations"]
                for ext in ("png", "pdf")
            ]
    return sorted(set(figures))


def prepare(eras, run):
    if run.exists():
        raise FileExistsError(f"Choose a new immutable run directory: {run}")
    preflight = read(CORE / "input_preflight.json")
    if preflight["passed"] is not True:
        raise ValueError("Input preflight has not passed")
    settings = read(CORE / "model.json")
    era_config = read(CORE / "eras.json")
    catalog = read(CORE / "samples.json")["eras"]
    sys.path.insert(0, str(CORE / "scripts/Jet_RecoilCorrection"))
    from samples import validate_catalog

    validate_catalog(catalog, eras)
    if settings["model"]["fit_regions"] != ["zee", "wcr"]:
        raise ValueError("This campaign requires ZEE+WCR")
    run.mkdir(parents=True)
    # The runtime is self-contained: definitions, metadata, tests and plot code are frozen together.
    shutil.copytree(
        CORE,
        run / "code_snapshot",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
    )
    hashes = {
        str(p.relative_to(run / "code_snapshot")): sha(p)
        for p in sorted((run / "code_snapshot").rglob("*"))
        if p.is_file()
    }
    tool_id = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    write(run / "code_sha256.json", hashes)
    with tarfile.open(run / "toolkit.tar.gz", "w:gz") as archive:
        archive.add(run / "code_snapshot", arcname=tool_id)
    write(
        run / "campaign.json",
        dict(
            eras=eras,
            fit_regions=["zee", "wcr"],
            validation_regions=["zee", "wcr", "zcr", "srsb"],
            baseline=settings["reference"],
            reference_changes=settings["reference_changes"],
            tool_id=tool_id,
            toolkit_sha256=sha(run / "toolkit.tar.gz"),
            core=str(CORE),
            max_jobs=1000,
            max_fits=1,
        ),
    )
    os.environ["RECOIL_RUN_DIR"] = str(run)
    os.environ["CROWN_JET_PT_MODE"] = "nominal"
    sys.path.insert(0, str(CORE / "scripts/Jet_RecoilCorrection"))
    import config

    settings["model"]["wcr_observation_slices"] = config.wcr_observation_slices(settings["model"])
    all_fit = []
    all_plot = []
    for era in eras:
        config.configure_reference(
            era,
            "inclusive",
            jet_pt_edges=list(map(float, settings["model"]["jet_pt_edges"])),
        )
        records = []
        for region in ("zee", "wcr", "zcr", "srsb"):
            for record in catalog[era][region]:
                for item in record["inputs"]:
                    stat = Path(item["path"]).stat()
                    if (
                        stat.st_size != item["size"]
                        or stat.st_mtime_ns != item["mtime_ns"]
                    ):
                        raise ValueError(f"Input changed: {item['path']}")
                records.append(dict(**record, era=era, region=region))
        fit_records = [r.copy() for r in records if r["region"] in ("zee", "wcr")]
        manifest = {
            **settings,
            "eras": [era],
            "jet_pt_mode": "nominal",
            "samples": fit_records,
            "luminosity_pb": era_config[era]["lumi_pb"],
            "input_adapter": era_config[era]["adapter"],
            "tool_id": tool_id,
            "shared_cache_dir": str(run / "shared"),
            "configuration": dict(
                selection=config.selection_metadata(),
                config_sha256=sha(CORE / "scripts/Jet_RecoilCorrection/config.py"),
            ),
        }
        write(run / era / "E/run_manifest.json", manifest)
        write(
            run / era / "E/validation_candidate/validation_manifest.json",
            {**manifest, "validation_samples": records},
        )
        all_fit.extend(fit_records)
        all_plot.extend(records)
    for i, r in enumerate(all_fit):
        r["id"] = i
    for i, r in enumerate(all_plot):
        r["id"] = i
    write(run / "inputs.json", dict(records=all_fit))
    write(run / "validation_inputs.json", dict(records=all_plot))
    write(run / "plots.json", dict(figures=figure_catalog(eras)))
    write(
        run / "completion.json",
        dict(completed=False, stage="prepared", tool_id=tool_id),
    )
    return all_fit, all_plot


def condor(eras, run, fit_records, plot_records, control):
    control.mkdir(parents=True, exist_ok=False)
    (control / "run.sh").write_text(f"""#!/usr/bin/env bash
set -e
source /cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc15-opt/setup.sh
source /cvmfs/sft.cern.ch/lcg/views/LCG_108_cuda/x86_64-el9-gcc13-opt/setup.sh
set -uo pipefail
export X509_USER_PROXY={PROXY}
export MPLCONFIGDIR=/tmp/recoil-mpl PYTHONPYCACHEPREFIX=/tmp/recoil-pycache
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CROWN_JET_PT_MODE=nominal
exec {PYTHON} -B "$@"
""")
    (control / "run.sh").chmod(0o755)
    (control / "stage.sub").write_text(f"""universe = vanilla
executable = {control}/run.sh
initialdir = {control}
should_transfer_files = NO
x509userproxy = {PROXY}
request_cpus = 10
request_memory = 30000MB
request_disk = 4000MB
+JobFlavour = "tomorrow"
MY.WantOS = "el9"
requirements = (Arch == "X86_64") && (TARGET.NODE_MOUNTS_AFS =?= true)
output = {control}/$(name).out
error = {control}/$(name).err
log = {control}/jobs.log
arguments = $(args)
queue 1
""")
    program = run / "code_snapshot/scripts/Jet_RecoilCorrection/stage_runner.py"
    lines = [
        f"NODE_STATUS_FILE {control}/pipeline.status 30 ALWAYS-UPDATE",
        "MAXJOBS FIT 1",
    ]
    nodes = []

    def node(name, stage, era=None, parents=(), index=None):
        memory, cpus = 30000, 10
        args = (
            f"{program} --run-dir {run} {stage}"
            + (f" --era {era}" if era else "")
            + (f" --index {index}" if index is not None else "")
        )
        lines.extend(
            [
                f"JOB {name} {control}/stage.sub",
                f'VARS {name} name="{name}" args="{args}" memory="{memory}" cpus="{cpus}"',
                f"RETRY {name} 1",
            ]
        )
        if parents:
            lines.append("PARENT " + " ".join(parents) + " CHILD " + name)
        if stage == "fit":
            lines.append(f"CATEGORY {name} FIT")
        nodes.append(
            dict(
                name=name,
                stage=stage,
                era=era,
                index=index,
                memory_mb=memory,
                cpus=cpus,
                parents=list(parents),
            )
        )

    metrics = []
    finished = []
    for era in eras:
        scans = []
        for i, r in enumerate(fit_records):
            if r["era"] != era:
                continue
            name = f"{era}_scan_{i}"
            scans.append(name)
            node(
                name,
                "scan",
                era,
                index=i,
            )
        pre = era + "_preflight"
        node(pre, "preflight", era, scans)
        fit = era + "_fit"
        node(fit, "fit", era, [pre])
        node(era + "_export", "export-sf", era, [fit])
        node(era + "_metrics", "metrics", era, [fit])
        metrics.append(era + "_metrics")
        finished.append(era + "_export")
    node("plot_setup", "plot-setup", parents=metrics)
    for era in eras:
        hist = []
        fit_hist = []
        ggf = []
        for i, r in enumerate(plot_records):
            if r["era"] != era:
                continue
            name = f"{era}_hist_{i}"
            node(name, "histogram", era, ["plot_setup"], index=i)
            hist.append(name)
            if r["region"] in ("zee", "wcr"):
                name = f"{era}_fit_hist_{i}"
                node(name, "fit-histogram", era, ["plot_setup"], index=i)
                fit_hist.append(name)
            else:
                name = f"{era}_ggf_hist_{i}"
                node(name, "ggf-histogram", era, [hist[-1]], index=i)
                ggf.append(name)
        agg = era + "_aggregate"
        node(agg, "aggregate", era, hist)
        fp = era + "_fit_plots"
        node(fp, "fit-plots", era, fit_hist)
        node(era + "_log_plots", "log-plots", era, [fp])
        node(era + "_extra", "extra", era, [agg])
        node(era + "_ggf_plots", "ggf-plots", era, [agg, *ggf])
        finished.extend(
            [agg, fp, era + "_log_plots", era + "_extra", era + "_ggf_plots"]
        )
    node("finish", "finish", parents=finished)
    (control / "pipeline.dag").write_text("\n".join(lines) + "\n")
    write(run / "jobs.json", dict(nodes=nodes, total=len(nodes)))
    subprocess.run(
        [
            "condor_submit_dag",
            "-no_submit",
            "-maxjobs",
            "1000",
            str(control / "pipeline.dag"),
        ],
        cwd=control,
        check=True,
    )
    return control / "pipeline.dag.condor.sub"


def replace_previous_run(run, control, eras):
    """Explicitly replace this named campaign after all its jobs have left the queue."""
    import re

    previous = read(run / "submission.json")
    campaign = read(run / "campaign.json")
    if campaign["eras"] != eras or Path(previous["control_dir"]).resolve() != control:
        raise ValueError(
            "Replacement must use the existing campaign eras and control directory"
        )
    cluster = int(re.search(r"cluster\s+(\d+)", previous["output"]).group(1))
    queue_text = subprocess.check_output(
        [
            "condor_q",
            "-constraint",
            f"ClusterId=={cluster} || DAGManJobId=={cluster}",
            "-json",
        ],
        text=True,
    )
    queue = json.loads(queue_text) if queue_text.strip() else []
    if queue:
        raise RuntimeError(
            "Old DAG or children are still in the queue; stop them before replacement"
        )
    catalog = read(CORE / "samples.json")["eras"]
    sys.path.insert(0, str(CORE / "scripts/Jet_RecoilCorrection"))
    from samples import validate_catalog

    validate_catalog(catalog, eras)
    for era in eras:
        for records in catalog[era].values():
            for record in records:
                for item in record["inputs"]:
                    if Path(item["path"]).resolve().is_relative_to(run):
                        raise ValueError(
                            "Refusing to replace a directory containing source ROOT inputs"
                        )
    old = dict(
        old_cluster=cluster,
        old_tool_id=campaign["tool_id"],
        reason="Restore process metadata and retain the leading 32 selected jets; full rescan",
        backup_created=False,
    )
    # Only the two explicitly named, previously submitted campaign directories are cleared.
    for directory in (run, control):
        for child in directory.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
        directory.rmdir()
    return old


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--era",
        nargs="+",
        choices=list(read(CORE / "eras.json")),
        default=list(read(CORE / "eras.json")),
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--control-dir", type=Path, required=True)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument(
        "--replace",
        action="store_true",
        help="Replace this stopped campaign in place without a backup",
    )
    a = p.parse_args()
    run = a.output.resolve()
    replacement = (
        replace_previous_run(run, a.control_dir.resolve(), a.era) if a.replace else None
    )
    fit, plot = prepare(a.era, run)
    if replacement:
        write(run / "rerun.json", replacement)
    submit = condor(a.era, run, fit, plot, a.control_dir.resolve())
    if a.prepare_only:
        print("PREPARED", run, submit)
        return
    result = subprocess.run(
        ["condor_submit", str(submit)],
        cwd=a.control_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    write(
        run / "submission.json",
        dict(
            output=result.stdout,
            control_dir=str(a.control_dir.resolve()),
            dag=str(a.control_dir.resolve() / "pipeline.dag"),
        ),
    )
    print(result.stdout)


if __name__ == "__main__":
    main()
