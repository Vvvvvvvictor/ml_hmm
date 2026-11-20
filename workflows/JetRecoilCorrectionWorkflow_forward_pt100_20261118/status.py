#!/usr/bin/env python3
"""Report master-DAG queue state and each era's fit/completion status."""

import argparse, json, re, subprocess
from collections import Counter
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--run-dir", type=Path, required=True)
a = p.parse_args()
run = a.run_dir
campaign = json.loads((run / "campaign.json").read_text())
submission = json.loads((run / "submission.json").read_text())
cluster = int(re.search(r"cluster\s+(\d+)", submission["output"]).group(1))
queue_text = subprocess.check_output(
    [
        "condor_q",
        *(["-name", submission["schedd"]] if submission.get("schedd") else []),
        "-constraint",
        f"ClusterId=={cluster} || DAGManJobId=={cluster}",
        "-json",
    ],
    text=True,
)
queue = json.loads(queue_text) if queue_text.strip() else []
labels = {
    1: "idle",
    2: "running",
    3: "removed",
    4: "completed",
    5: "held",
    6: "transferring",
    7: "suspended",
}
dag_status = {}
nodes = []
status_file = Path(submission.get("status_file", str(Path(submission["control_dir"]) / "pipeline.status")))
if status_file.exists():
    content = status_file.read_text()
    for key in ("NodesTotal", "NodesDone", "NodesQueued", "NodesFailed", "NodesFutile"):
        match = re.search(r"\b" + key + r" = (\d+)", content)
        if match:
            dag_status[key] = int(match.group(1))
    for block in re.findall(r"\[\s*(.*?)\s*\]", content, re.S):
        name = re.search(r'Node = "([^"]+)"', block)
        status = re.search(r"NodeStatus = (\d+)", block)
        if name and status:
            nodes.append((name.group(1), int(status.group(1))))
dag_status["failed_nodes"] = [name for name, status in nodes if status == 6]
fits = {}
for era in campaign["eras"]:
    path = run / era / "E/fit/fit_results.json"
    if path.exists():
        f = json.loads(path.read_text())
        fits[era] = {
            k: f[k] for k in ("optimizer_converged", "errors_valid", "objective")
        }
        fits[era]["boundary_parameters"] = len(f.get("boundary_parameter_indices", []))
        fits[era]["covariance_method"] = f.get("covariance_method")
    else:
        failed = [
            name
            for name, status in nodes
            if name.startswith(era + "_") and status in (6, 7)
        ]
        scans = [
            (name, status) for name, status in nodes if name.startswith(era + "_scan_")
        ]
        fits[era] = dict(
            status="blocked" if failed else "pending",
            scans_done=sum(status == 5 for _, status in scans),
            scans_total=len(scans),
        )
print(
    json.dumps(
        dict(
            cluster=cluster,
            schedd=submission.get("schedd"),
            control_dir=submission["control_dir"],
            queue=dict(Counter(labels[x["JobStatus"]] for x in queue)),
            fits=fits,
            dag=dag_status,
            completion=json.loads((run / "completion.json").read_text()),
        ),
        indent=2,
    )
)
