"""Numerical regression against the archived C-disabled model and ROOT schema smoke checks."""

import argparse, importlib.util, json, os, sys, tempfile
from pathlib import Path
import numpy as np

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE / "scripts/Jet_RecoilCorrection"))
parser = argparse.ArgumentParser()
parser.add_argument("--reference-core", required=True, type=Path)
a = parser.parse_args()
run = Path(tempfile.mkdtemp(prefix="recoil_verify_"))
(run / "campaign.json").write_text(
    json.dumps(dict(eras=["2024"], validation_regions=["zee", "wcr", "zcr", "srsb"]))
)
os.environ["RECOIL_RUN_DIR"] = str(run)
import config

model = json.loads((CORE / "model.json").read_text())
config.configure_reference(
    "2024", jet_pt_edges=list(map(float, model["model"]["jet_pt_edges"]))
)
import conditional_model as new
import w_sf

old_w_name = sys.modules["w_sf"]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sys.modules["w_sf"] = module(
    "reference_w_sf", a.reference_core / "scripts/Jet_RecoilCorrection/w_sf.py"
)
old = module(
    "reference_model",
    a.reference_core / "scripts/Jet_RecoilCorrection/conditional_model.py",
)
sys.modules["w_sf"] = old_w_name
rng = np.random.default_rng(794)
checks = []
for region in ["zcr", "wcr"]:
    n = 90
    keys = np.full((n, 117), 65535, np.uint16)
    njet = rng.integers(0, 5, n)
    keys[:, 1] = njet
    keys[:, 2] = rng.integers(0, 130, n)
    keys[:, 3] = np.arange(n) % 2
    keys[:, 69] = rng.integers(0, 4, n)
    keys[:, 105] = 0
    keys[:, 112] = rng.integers(0, 10, n)
    keys[:, 116] = rng.integers(0, 26, n)
    for i, k in enumerate(njet):
        keys[i, 4 : 4 + k] = rng.integers(0, 20, k)
        keys[i, 37 : 37 + k] = rng.integers(0, 8, k)
    w = rng.normal(1, 1.5, n)
    sums = np.column_stack([w, w * w, abs(w), np.ones(n), w > 0, w < 0, abs(w)])
    families = np.arange(n) % 2
    args = (keys, sums, families, 8, 20, region, ["background", "DY"], 130)
    import inspect
    extra = {"grouping_columns": (105,)} if "grouping_columns" in inspect.signature(old.design).parameters else {}
    before = old.design(*args, use_w_sf=region == "wcr", muon_bins=10, **extra)
    if extra:
        before["c_enabled"] = np.zeros(len(before["group_labels"]), bool)
    after = new.design(*args, use_w_sf=region == "wcr", muon_bins=10)
    x = rng.normal(0, 0.12, 8 + 130 + (104 if region == "wcr" else 0))
    pred, jac = new.model_and_jacobian(x, after)
    pred0, jac0 = old.model_and_jacobian(x, before)
    np.testing.assert_allclose(pred, pred0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(jac, jac0, rtol=1e-12, atol=1e-12)
    cov = lambda m: (m["h"].T @ m["h"].multiply(sums[:, 1, None])).toarray()
    np.testing.assert_array_equal(cov(after), cov(before))
    for i in range(len(x)):
        hi = x.copy()
        lo = x.copy()
        hi[i] += 1e-6
        lo[i] -= 1e-6
        derivative = (
            new.model_and_jacobian(hi, after)[0] - new.model_and_jacobian(lo, after)[0]
        ) / 2e-6
        np.testing.assert_allclose(derivative, jac[:, i], rtol=1e-6, atol=1e-7)
    checks.append(
        region
        + " signed weights / repeated jets / zero jets / Jacobian / covariance regression"
    )
from fit_model import sf_prior_rows, signatures, SignatureAccumulator, combine, inapplicable_candidates

# Repeated signed-weight signatures must retain all moments and maxima under a different merge tree.
k = np.tile(keys[:30], (20, 1))
w = rng.normal(size=len(k))
v = np.column_stack([w, w * w, abs(w), np.ones(len(w)), w > 0, w < 0, abs(w)])
acc = SignatureAccumulator()
for i in range(0, len(k), 17):
    acc.add((k[i : i + 17], v[i : i + 17]))
actual, expected = acc.finish(), combine([(k, v)])
np.testing.assert_array_equal(actual[0], expected[0])
np.testing.assert_allclose(actual[1], expected[1], rtol=1e-12, atol=1e-12)
checks.append("Balanced scan accumulation preserves signed moments and maximum weights")
k = config.N_JET_CELLS
total = k + config.N_DY_CELLS + w_sf.N_W_SF
manifest = dict(model=model["model"], configuration={"selection": config.selection_metadata()})
eligible = np.ones(total, bool)
eligible[inapplicable_candidates(manifest)] = False
r, j = sf_prior_rows(
    np.zeros(total), model["fit_priors"],
    dict(jet_matched=[0, k // 2], jet_unmatched=[k // 2, k]), eligible,
)
assert r.shape == (int(eligible[:k].sum()),) and j.shape == (len(r), total) and not r.any()
checks.append(f"{len(r)} free Gaussian prior rows retained")
from until import normalization_sumw


class Branch:
    def __init__(self, values):
        self.values = values

    def array(self, library):
        return np.asarray(self.values)


for adapter, name in [("native", "genEventSumw"), ("sampled", "normalization_sumw")]:
    root = {"conditions": {name: Branch([5.0, -2.0])}}
    assert normalization_sumw(root, adapter) == 3.0
    for bad in ([1.0, -1.0], [np.nan]):
        root["conditions"][name] = Branch(bad)
        try:
            normalization_sumw(root, adapter)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid denominator accepted")
    try:
        normalization_sumw({"conditions": {}}, adapter)
    except KeyError:
        pass
    else:
        raise AssertionError("Missing denominator accepted")
checks.append(
    "Native/sampled strict normalization, negative weights, zero and missing denominator"
)
from plots import agreement_scales

observed = np.array([2.0, 4.0])
pred = {"nominal": np.array([1.0, 2.0]), "corrected": np.array([2.0, 2.0])}
assert agreement_scales(observed, pred, "physical") == {
    "nominal": 1.0,
    "corrected": 1.0,
}
assert agreement_scales(observed, pred, "absolute") == {
    "nominal": 1.0,
    "corrected": 1.0,
}
assert agreement_scales(observed, pred, "shape") == {"nominal": 2.0, "corrected": 1.5}
checks.append(
    "Three display normalizations leave predictions and fit kernels unchanged"
)
from sf_export import export_sf, verify_export, build_payload
import shutil

base = Path(
    "/eos/user/j/jiehan/HMuMuShare/JetRecoilCorrection_Run3/20261111_641e_forward_pt70/2024/E"
)
fixture = run / "export"
(fixture / "fit").mkdir(parents=True)
for source, dest in [
    (base / "run_manifest.json", fixture / "run_manifest.json"),
    (base / "fit/fit_results.json", fixture / "fit/fit_results.json"),
]:
    shutil.copyfile(source, dest)
# The archived result predates explicit validation-region metadata. Upgrade only the test fixture.
fixture_fit = json.loads((fixture / "fit/fit_results.json").read_text())
fixture_fit["validation_regions"] = ["zcr", "wcr", "srsb"]
(fixture / "fit/fit_results.json").write_text(json.dumps(fixture_fit))
# Upgrade only the temporary archived fixture to the current uncertainty interface.
fixture_manifest=json.loads((fixture / "run_manifest.json").read_text())
import hashlib
fixture_manifest["tool_id"]="archived-fixture-" + hashlib.sha256((base / "run_manifest.json").read_bytes()).hexdigest()
(fixture / "run_manifest.json").write_text(json.dumps(fixture_manifest))
for name in ("parameters.npy", "hessian.npy", "data_hessian.npy"):
    shutil.copyfile(base / "fit" / name, fixture / "fit" / name)
from sf_covariance import recompute
recompute(fixture / "fit")
export_sf(fixture)
verify_export(fixture)
m = json.loads((fixture / "run_manifest.json").read_text())
f = json.loads((fixture / "fit/fit_results.json").read_text())
assert len(build_payload(m, f)["corrections"]) == 4
f["errors_valid"] = False
try:
    build_payload(m, f)
except ValueError:
    pass
else:
    raise AssertionError("Invalid fit was exported")
checks.append(
    "JSON/gzip schema-v2 every-cell C++ evaluator at rtol 1e-14 and invalid-fit rejection"
)
print(json.dumps(dict(passed=True, checks=checks), indent=2))
