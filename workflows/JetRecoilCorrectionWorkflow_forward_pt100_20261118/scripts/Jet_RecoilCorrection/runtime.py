"""Shared production paths and JSON/NPZ I/O."""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
import numpy as np

RUN = Path(os.environ["RECOIL_RUN_DIR"])
_campaign = json.loads((RUN / "campaign.json").read_text())
ERAS = tuple(_campaign["eras"])
REGIONS = tuple(_campaign["validation_regions"])
PYTHON = "/eos/user/j/jiehan/ml-gpu-env/bin/python3"
PROXY = "/afs/cern.ch/user/j/jiehan/x509up_u152815"
SOURCE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(p)


def save(path, **values):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pending = p.with_suffix(".npz.partial")
    # ZIP headers are rewritten during compression. Finish those writes on local disk.
    with tempfile.TemporaryFile(dir=os.environ.get("_CONDOR_SCRATCH_DIR")) as archive:
        np.savez_compressed(archive, **values)
        archive.seek(0)
        with pending.open("wb") as out:
            shutil.copyfileobj(archive, out)
    pending.replace(p)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
