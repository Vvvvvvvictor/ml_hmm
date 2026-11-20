#!/usr/bin/env python3
"""Submit fits and standard Data/MC agreement plots from a fit-ready campaign."""

import argparse
from pathlib import Path
import re
import shutil
import subprocess

from recover_fit_inputs import campaign_identity, read, write


def finish(run):
    campaign = read(run / "campaign.json")
    results = {}
    lines = ["# ZEE + WCR fit and Data/MC agreement", ""]
    for era in campaign["eras"]:
        root = run / era / "E"
        fit = read(root / "fit/fit_results.json")
        if not all(fit[key] for key in ("optimizer_converged", "fit_success", "errors_valid")):
            raise ValueError(f"Unaccepted fit for {era}")
        root_plots = root / "validation_candidate" / era
        paths = {
            region: root_plots / region / "agreement_summary.pdf"
            for region in campaign["validation_regions"]
        }
        for path in paths.values():
            with path.open("rb") as f:
                if f.read(5) != b"%PDF-" or path.stat().st_size < 1000:
                    raise ValueError(f"Missing or invalid agreement PDF: {path}")
        metrics = run / "validation_metrics" / f"{era}.json"
        assert set(read(metrics)) == set(paths)
        results[era] = dict(
            fit=str((root / "fit/fit_results.json").relative_to(run)),
            corrections=str((root / "fit/corrections.json.gz").relative_to(run)),
            agreement={key: str(path.relative_to(run)) for key, path in paths.items()},
        )
        entry = results[era]
        links = ", ".join(f"[{region.upper()}]({path})" for region, path in entry["agreement"].items())
        lines.append(f"- {era}: [fit]({entry['fit']}), [SF]({entry['corrections']}), {links}")
    (run / "FIT_AGREEMENT.md").write_text("\n".join(lines) + "\n")
    write(run / "fit_agreement_completion.json", dict(completed=True, results=results))


def prepare(run, control):
    ready = read(run / "fit_inputs_completion.json")
    campaign = read(run / "campaign.json")
    eras = campaign["eras"]
    records = read(run / "validation_inputs.json")["records"]
    if (ready["fit_inputs_ready"] is not True or ready["tool_id"] != campaign_identity(run)
            or ready["eras"] != eras
            or ready["samples"] != len(read(run / "inputs.json")["records"])):
        raise ValueError("Fit input readiness does not match campaign")

    control.mkdir()
    previous = Path(ready["control_dir"])
    shutil.copy2(previous / "run.sh", control / "run.sh")
    for script in (Path(__file__), Path(__file__).with_name("recover_fit_inputs.py")):
        shutil.copy2(script, control / script.name)
    sub = (previous / "stage.sub").read_text().replace(str(previous), str(control))
    (control / "stage.sub").write_text(sub.replace("request_memory = 30000MB", "request_memory = $(memory)MB"))
    default = run / "code_snapshot/scripts/Jet_RecoilCorrection"
    lines = [
        f"NODE_STATUS_FILE {control}/pipeline.status 30 ALWAYS-UPDATE",
        f"MAXJOBS FIT {campaign['max_fits']}",
    ]

    def node(name, arguments, *, parents=(), memory=30000, fit=False):
        lines.extend([
            f"JOB {name} {control}/stage.sub",
            f'VARS {name} name="{name}" args="{arguments}" memory="{memory}"',
            f"RETRY {name} 1",
        ])
        if parents:
            lines.append(f"PARENT {' '.join(parents)} CHILD {name}")
        if fit:
            lines.append(f"CATEGORY {name} FIT")

    stage = f"{default}/stage_runner.py --run-dir {run}"
    for era in eras:
        memory = 48000 if era in ("2024", "2025") else 30000
        node(f"{era}_fit", f"{stage} fit --era {era}", memory=memory, fit=True)
        node(f"{era}_export", f"{stage} export-sf --era {era}", parents=[f"{era}_fit"])
        node(f"{era}_metrics", f"{stage} metrics --era {era}", parents=[f"{era}_fit"], memory=memory)
        for region in campaign["validation_regions"]:
            (run / era / "E/validation_candidate" / era / "raw" / region).mkdir(parents=True, exist_ok=True)
    (run / "metrics").mkdir(exist_ok=True)
    (run / "validation_metrics").mkdir(exist_ok=True)
    node("plot_setup", f"{stage} plot-setup", parents=[f"{era}_fit" for era in eras])
    hist = {era: [] for era in eras}
    for i, record in enumerate(records):
        era = record["era"]
        name = f"{era}_hist_{i}"
        # The frozen histogram stage checks input identities and writes atomic
        # output files. Aggregate checks record identity and fitted SF digest.
        node(name, f"{default}/campaign.py --run-dir {run} histogram --index {i}",
             parents=["plot_setup"])
        hist[era].append(name)
    for era in eras:
        node(f"{era}_agreement", f"{stage} aggregate --era {era}",
             parents=[f"{era}_metrics", *hist[era]])
    node("fit_agreement_ready", f"{control}/{Path(__file__).name} {run} --finish",
         parents=[f"{era}_{action}" for era in eras for action in ("export", "agreement")])
    (control / "pipeline.dag").write_text("\n".join(lines) + "\n")
    subprocess.run(
        ["condor_submit_dag", "-no_submit", "-maxjobs", str(campaign["max_jobs"]), "pipeline.dag"],
        cwd=control, check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--finish", action="store_true")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    if args.finish:
        finish(run)
        return
    control = args.control.resolve()
    if args.submit:
        if (control / "submission.json").exists():
            raise FileExistsError("Already submitted")
        output = subprocess.check_output(["condor_submit", "pipeline.dag.condor.sub"], cwd=control, text=True)
        print(output, end="")
        submission = dict(cluster=int(re.search(r"cluster (\d+)", output)[1]), control_dir=str(control))
        write(control / "submission.json", submission)
        write(run / "fit_agreement_submission.json", submission)
    else:
        prepare(run, control)


if __name__ == "__main__":
    main()
