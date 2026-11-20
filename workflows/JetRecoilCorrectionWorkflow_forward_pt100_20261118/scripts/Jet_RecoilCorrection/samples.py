"""Strict, report-derived nominal sample catalog. No directory discovery."""

import json
from copy import deepcopy
from pathlib import Path
import config

CATALOG = json.loads(
    (Path(__file__).resolve().parents[2] / "samples.json").read_text()
)["eras"]


def sample_records(region, era=None):
    return deepcopy(CATALOG[config.ERA if era is None else era][region])


def input_record(path, region):
    target = str(Path(path))
    matches = [
        r
        for r in CATALOG[config.ERA][region]
        if any(i["path"] == target for i in r["inputs"])
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Input is not uniquely declared: {config.ERA}/{region}/{path}"
        )
    return matches[0]


def validate_catalog(catalog, eras):
    """Reject incomplete or inconsistent records before preparing any campaign."""
    from sample_groups import process_family

    required = {
        "sample",
        "group",
        "process",
        "cross_section_pb",
        "inputs",
        "production_fingerprint",
        "dy_sf_applicable",
    }
    count = 0
    for era in eras:
        for region in ("zee", "wcr", "zcr", "srsb"):
            records = catalog[era][region]
            if sum(r["sample"] == "data" for r in records) != 1:
                raise ValueError(f"Expected exactly one data record: {era}/{region}")
            for r in records:
                missing = required - r.keys()
                if missing:
                    raise ValueError(
                        f"{era}/{region}/{r['sample']}: missing fields {sorted(missing)}"
                    )
                expected = process_family(r["sample"], r["group"])
                if r["process"] != expected:
                    raise ValueError(
                        f"Incorrect process for {era}/{region}/{r['sample']}: expected {expected}"
                    )
                if not r["inputs"]:
                    raise ValueError(f"Empty input list: {era}/{region}/{r['sample']}")
                count += 1
    return count
