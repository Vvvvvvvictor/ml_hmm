#!/usr/bin/env python3
"""Check every report-listed file and freeze its schema, provenance and identity."""

import argparse, hashlib, json, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np

CORE = Path(__file__).resolve().parent
sys.path.insert(0, str(CORE / "scripts/Jet_RecoilCorrection"))
import config
from root_io import open_root, jet_tree
from until import normalization_sumw, selection, event_weights


def inspect(task):
    era, region, record, item = task
    path = Path(item["path"])
    stat = path.stat()
    data = record["sample"] == "data"
    required = set(config.region_branches(region))
    if not data:
        required.update(("Jet_isGenMatched", *config.weight_branches(region)))
    with open_root(path) as root:
        logical = jet_tree(root[config.TREE])
        missing = required - set(logical.keys())
        if missing:
            raise KeyError(f"{era}/{region}/{path.name}: missing {sorted(missing)}")
        if logical.num_entries != item["expected_entries"]:
            raise ValueError(f"Event count differs from report: {path}")
        meta = {}
        for name in (
            "production_fingerprint",
            "output_schema",
            "matching_definition",
            "corrections_manifest",
        ):
            value = str(root[name])
            actual = hashlib.sha256(value.encode()).hexdigest()
            if actual != item["provenance_sha256"][name]:
                raise ValueError(f"Changed {name}: {path}")
            meta[name] = value
        if meta["production_fingerprint"] != record["production_fingerprint"]:
            raise ValueError(f"Wrong production: {path}")
        adapter = config.ERAS[era]["adapter"]
        if adapter == "sampled":
            value = str(root["era_sampling"])
            if (
                hashlib.sha256(value.encode()).hexdigest()
                != item["provenance_sha256"]["era_sampling"]
            ):
                raise ValueError(f"Changed sampling {path}")
            meta["era_sampling"] = json.loads(value)
        denominator = None if data else normalization_sumw(root, adapter)
        a = logical.arrays(sorted(required), entry_stop=256, library="ak")
        selected = a[selection(a, region)]
        scale = (
            1.0
            if data
            else config.ERAS[era]["lumi_pb"] * record["cross_section_pb"] / denominator
        )
        w = event_weights(selected, scale, region, data)
        schema = hashlib.sha256(
            json.dumps(sorted(root[config.TREE].typenames().items())).encode()
        ).hexdigest()
        return {
            **item,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "schema_sha256": schema,
            "normalization_sumw": denominator,
            **meta,
        }, dict(
            era=era,
            region=region,
            sample=record["sample"],
            path=str(path),
            entries=logical.num_entries,
            smoke_entries=len(a),
            smoke_selected=len(selected),
            smoke_negative=int((w < 0).sum()),
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    catalog = json.loads((CORE / "samples.json").read_text())
    tasks = []
    for era, regions in catalog["eras"].items():
        for region, records in regions.items():
            for r in records:
                for item in r["inputs"]:
                    tasks.append((era, region, r, item))
    reports = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(inspect, t): t for t in tasks}
        for f in as_completed(pending):
            t = pending[f]
            new, report = f.result()
            payload = new.pop("corrections_manifest")
            digest = hashlib.sha256(payload.encode()).hexdigest()
            new["corrections_manifest_sha256"] = digest
            target = CORE / "provenance" / (digest + ".json")
            target.parent.mkdir(exist_ok=True)
            if not target.exists():
                target.write_text(payload)
            t[3].update(new)
            reports.append(report)
            print(
                "PREFLIGHT",
                len(reports),
                len(tasks),
                report["era"],
                report["region"],
                report["sample"],
                flush=True,
            )
    (CORE / "samples.json").write_text(json.dumps(catalog, indent=2) + "\n")
    (CORE / "input_preflight.json").write_text(
        json.dumps(dict(passed=True, files=len(reports), records=reports), indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
