"""Regression for process metadata, leading-32 jets, and actual seven-era cache preflight."""

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import awkward as ak
import numpy as np

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE / "scripts/Jet_RecoilCorrection"))
from samples import CATALOG, validate_catalog
from sample_groups import process_family
import config
from until import selected_jets, selection

ERAS = list(config.ERAS)
assert validate_catalog(CATALOG, ERAS) == 376
from copy import deepcopy

bad = deepcopy(CATALOG)
bad["2022"]["zee"][0].pop("process")
try:
    validate_catalog(bad, ERAS)
except ValueError as e:
    assert "process" in str(e)
else:
    raise AssertionError("Missing process was accepted")
counts = [0, 1, 31, 32, 33, 40]
rows = []
expected = []
for count in counts:
    # Deliberately unsorted pT, tied pT, and two failing jets, to check the common mask and stable ordering.
    pt = np.r_[30 + np.arange(count) % 7, 20.0, 200.0]
    eta = np.r_[np.linspace(-2, 2, count), 0.0, 5.0]
    phi = np.arange(count + 2) * 0.03
    keep = np.flatnonzero((pt > 25) & (np.abs(eta) < 4.7))
    order = keep[np.argsort(-pt[keep], kind="stable")[:32]]
    rows.append(
        dict(
            Jet_corrPt=pt,
            Jet_corrEta=eta,
            Jet_corrPhi=phi,
            Jet_corrMass=np.arange(count + 2) + 1.0,
            Jet_isGenMatched=np.arange(count + 2) % 2,
        )
    )
    expected.append(order)
a = ak.Array(rows)
for name, values in {
    "single_muon_pt": 50.0,
    "single_muon_phi": 0.4,
    "met_pt": 40.0,
    "met_phi": 2.0,
    "muon_met_mt": 60.0,
    "single_muon_trigger_flag": 1,
    "nmuons": 1,
    "nelectrons": 0,
    "nbjets_medium": 0,
    "nbjets_loose": 0,
    "zee_pt": 12.0,
    "zee_phi": 0.2,
    "zee_mass": 90.0,
}.items():
    a = ak.with_field(a, np.full(len(a), values), name)
manual = a
for name in rows[0]:
    manual = ak.with_field(
        manual,
        ak.Array([np.asarray(a[name][i])[idx] for i, idx in enumerate(expected)]),
        name,
    )
with tempfile.TemporaryDirectory(prefix="recoil_repair_", dir="/tmp") as temp:
    run = Path(temp)
    (run / "campaign.json").write_text(
        json.dumps(dict(eras=ERAS, validation_regions=["zee", "wcr", "zcr", "srsb"]))
    )
    os.environ["RECOIL_RUN_DIR"] = str(run)
    import runtime, campaign, root_io

    for name in ("RUN", "ERAS", "read", "write", "save", "sha"):
        setattr(campaign, name, getattr(runtime, name))
    from fit_model import signatures
    from histograms import _jets
    from validation_plots import event_factors

    config.configure_reference("2024")
    stats = {}
    jets = selected_jets(a, False, statistics=stats)
    np.testing.assert_array_equal(ak.to_numpy(ak.num(jets[0])), np.minimum(counts, 32))
    assert stats == dict(truncated_events=2, discarded_jets=9, max_selected_jets=40)
    for k, name in enumerate(
        ("Jet_corrPt", "Jet_corrEta", "Jet_corrPhi", "Jet_corrMass")
    ):
        assert ak.to_list(jets[k]) == ak.to_list(manual[name])
    decoded = 1 - (jets[4] // (config.N_PT * config.N_ETA))
    assert ak.to_list(decoded) == ak.to_list(manual.Jet_isGenMatched)
    np.testing.assert_array_equal(selection(a, "wcr"), selection(manual, "wcr"))
    for region in ("zee", "wcr"):
        for data in (False, True):
            k = signatures(a, region, False, is_data=data)
            np.testing.assert_array_equal(
                k, signatures(manual, region, False, is_data=data)
            )
            np.testing.assert_array_equal(k[:, 1], np.minimum(counts, 32))
    fake = dict(jet_sf=np.full((2, 7, 40), 1.01), w_sf=np.full(104, 1.1))
    factors = event_factors(
        _jets(a, False),
        dict(sample="W", group="W", region="wcr", dy_sf_applicable=False),
        fake,
        mumet=np.full(len(a), 40.0),
    )
    np.testing.assert_allclose(
        factors[:, 1], 1.01 ** np.minimum(counts, 32), rtol=1e-12
    )
    print(
        "TOY leading32 cases, alignment, event selection, signatures and SF product PASSED",
        flush=True,
    )
    # Read the exact 20k-entry interval in which the old 2026 WCR data scan failed.
    config.configure_reference("2026")
    r = next(r for r in CATALOG["2026"]["wcr"] if r["sample"] == "data")
    original = next(
        root_io.input_batches(
            Path(r["inputs"][0]["path"]),
            "wcr",
            True,
            config.region_branches("wcr"),
            entry_start=1_380_000,
            entry_stop=1_400_000,
        )
    )[0]
    selected = original[selection(original, "wcr")]
    real_stats = {}
    keys = signatures(selected, "wcr", False, is_data=True, statistics=real_stats)
    assert real_stats["truncated_events"] > 0, real_stats
    assert len(keys) == len(selected) and keys[:, 1].max() <= 32
    print(
        "REAL 2026 WCR formerly failing interval",
        len(original),
        len(selected),
        real_stats,
        flush=True,
    )
    # Compare ordinary <=32 event signatures to the prior implementation, without copying old outputs.
    oldpath = Path(
        "/eos/user/j/jiehan/HMuMuShare/JetRecoilCorrectionWorkflow_backup_20261101T143656Z/scripts/Jet_RecoilCorrection/fit_model.py"
    )
    spec = importlib.util.spec_from_file_location("old_fit_repair_reference", oldpath)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    safe = selected[ak.num(selected.Jet_corrPt) <= 32]
    np.testing.assert_array_equal(
        signatures(safe, "wcr", False, is_data=True),
        old.signatures(safe, "wcr", False, is_data=True),
    )
    try:
        old.signatures(selected, "wcr", False, is_data=True)
    except ValueError as e:
        assert "MAX_JETS" in str(e)
    else:
        raise AssertionError("Old failure interval did not reproduce")
    print("OLD failure reproduced; <=32 signatures unchanged PASSED", flush=True)
    settings = json.loads((CORE / "model.json").read_text())
    fit_records = []
    for era in ERAS:
        config.configure_reference(era)
        records = [
            dict(**r, era=era, region=region)
            for region in ("zee", "wcr")
            for r in CATALOG[era][region]
        ]
        for r in records:
            r["id"] = len(fit_records)
            fit_records.append(r)
        manifest = {
            **settings,
            "eras": [era],
            "samples": records,
            "shared_cache_dir": str(run / "shared"),
            "jet_pt_mode": "nominal",
            "configuration": dict(
                selection=config.selection_metadata(),
                config_sha256=hashlib.sha256(
                    (CORE / "scripts/Jet_RecoilCorrection/config.py").read_bytes()
                ).hexdigest(),
            ),
        }
        runtime.write(run / era / "E/run_manifest.json", manifest)
    runtime.write(run / "inputs.json", dict(records=fit_records))
    # A bounded test-only reader; production retains max_events=0 and uses the unmodified reader.
    reader = root_io.input_batches

    def bounded(*args, **kwargs):
        yield from reader(*args, **kwargs, entry_stop=256)

    root_io.input_batches = bounded
    for era in ERAS:
        config.configure_reference(era)
        indices = [i for i, r in enumerate(fit_records) if r["era"] == era]
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(campaign.scan, indices))
        campaign.preflight(era)
        print("REAL CACHE/PREFLIGHT PASSED", era, flush=True)
    root_io.input_batches = reader
    print(
        json.dumps(
            dict(
                passed=True,
                process_records=376,
                cases=counts,
                real_failure_interval=real_stats,
                real_sample_scans=len(fit_records),
                preflight_eras=ERAS,
                temporary_products_removed_on_exit=True,
            )
        ),
        flush=True,
    )
