"""Process labels and atomic JSON output shared by the fit."""

import json

from pathlib import Path

import config


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def process_family(sample, group):
    if group == "EWK":
        return "EWK_W" if sample.startswith("Wto") else "EWK_Z"
    if group == "Top":
        return "TT" if sample.startswith("TT") else "tW"
    if group in ("Diboson", "Triboson"):
        return sample.split("to")[0].split("_")[0]
    return group
