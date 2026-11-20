"""Scale-invariant local Gauss-Newton errors; covariance includes SF priors."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy import linalg

METHOD = "normalized_gauss_newton_v2"
NOTE = ("Local Gaussian-prior-regularized Gauss-Newton absolute errors; "
        "not calibrated confidence intervals near SF=0. Propagate using full covariance_sf.npy.")


def information_covariance(hessian, eligible):
    h = np.asarray(hessian, dtype=float)
    eligible = np.asarray(eligible, dtype=bool)
    if h.shape != (len(eligible), len(eligible)) or not np.isfinite(h).all():
        raise ValueError("Invalid or nonfinite Hessian")
    h = (h + h.T) * 0.5
    diagonal = np.diag(h)
    if np.any(diagonal < 0):
        raise ValueError("Negative Hessian diagonal")
    active = eligible & (diagonal > 0)
    # A PSD matrix with a zero diagonal must have a zero row/column.
    if np.any(h[diagonal == 0] != 0):
        raise ValueError("Zero diagonal with nonzero Hessian row")
    norms = np.sqrt(diagonal[active])
    a = h[np.ix_(active, active)] / norms[:, None] / norms[None, :]
    eig, vectors = linalg.eigh((a + a.T) * 0.5)
    threshold = max(float(eig.max(initial=0)) * 1e-10, 1e-12)
    if np.any(eig < -threshold):
        raise ValueError("Hessian is not positive semidefinite")
    measured = eig > threshold
    covariance = np.zeros_like(h)
    covariance[np.ix_(active, active)] = ((vectors[:, measured] / eig[measured])
        @ vectors[:, measured].T / norms[:, None] / norms[None, :])
    identifiable = np.zeros(len(eligible), dtype=bool)
    # Null-space projection is tested in standardized coordinates, not log-SF units.
    identifiable[active] = np.sum(vectors[:, ~measured] ** 2, axis=1) < 1e-10
    return dict(covariance=covariance, active=active, eig=eig, vectors=vectors,
                measured=measured, identifiable=identifiable)


def propagate_sf_uncertainty(gradient, covariance_sf):
    """Absolute local error of one quantity, retaining all SF cross terms."""
    g = np.asarray(gradient, dtype=float)
    c = np.asarray(covariance_sf, dtype=float)
    if g.ndim != 1 or c.shape != (g.size, g.size) or not np.isfinite(g).all() or not np.isfinite(c).all():
        raise ValueError("Invalid SF propagation inputs")
    if not np.allclose(c, c.T, rtol=1e-12, atol=1e-15):
        raise ValueError("Nonsymmetric SF covariance")
    variance = float(g @ c @ g)
    tolerance = 1e-12 * float(np.abs(g) @ np.abs(c) @ np.abs(g))
    if not np.isfinite(variance) or variance < -tolerance:
        raise ValueError("Invalid propagated SF variance")
    return float(np.sqrt(max(variance, 0)))


def finite_list(array):
    a = np.asarray(array)
    return np.where(np.isfinite(a), a, None).tolist()


def update_payload(payload, parameters, total, data):
    sf = np.exp(parameters)
    if np.any(~np.isfinite(sf) | (sf <= 0)):
        raise ValueError("Nonpositive or nonfinite SF")
    for name in ("jet", "dy", "w"):
        field = name + "_sf"
        if field in payload:
            lo, hi = payload["parameter_slices"][name]
            saved = np.asarray(payload[field], dtype=float).ravel()
            if not np.allclose(saved, sf[lo:hi], rtol=2e-14, atol=0):
                raise ValueError("Saved central SF does not match parameters: " + name)
            sf[lo:hi] = saved  # Preserve original serialization, including last-bit exp differences.
    cov = total["covariance"]
    covariance_sf = sf[:, None] * cov * sf[None, :]
    if not np.isfinite(covariance_sf).all():
        raise ValueError("Nonfinite SF covariance")
    error = np.sqrt(np.maximum(np.diag(covariance_sf), 0))
    identifiable = total["identifiable"]
    error[~identifiable] = np.nan
    for name in ("jet", "dy", "w"):
        field = name + "_sf"
        if field not in payload:
            continue
        lo, hi = payload["parameter_slices"][name]
        shape = np.asarray(payload[field]).shape
        payload[field + "_error"] = finite_list(error[lo:hi].reshape(shape))
        payload[field + "_error_valid"] = identifiable[lo:hi].reshape(shape).tolist()
        payload[field + "_near_boundary"] = ((sf[lo:hi] - error[lo:hi]) < 0).reshape(shape).tolist()
    eligible = np.ones(len(sf), dtype=bool)
    eligible[payload["inapplicable_parameters"]] = False
    valid = bool(payload["optimizer_converged"] and total["measured"].sum() == eligible.sum()
                 and identifiable[eligible].all() and np.isfinite(error[eligible]).all())
    payload.update(errors_valid=valid, regularized_errors_valid=valid, fit_success=valid,
        regularized_rank=int(total["measured"].sum()), rank=int(data["measured"].sum()),
        data_rank=int(data["measured"].sum()), active_parameters=int(total["active"].sum()),
        identifiable=bool(data["measured"].sum() == eligible.sum()),
        data_identifiable=bool(data["measured"].sum() == eligible.sum()),
        unsupported_parameters=np.flatnonzero(~total["active"] & eligible).tolist(),
        weak_directions=int((~total["measured"]).sum()),
        data_weak_directions=int((~data["measured"]).sum()),
        total_information_eigenvalues=total["eig"].tolist(),
        data_information_eigenvalues=data["eig"].tolist(),
        covariance_method=METHOD, uncertainty_note=NOTE,
        uncertainty_algorithm_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        covariance_coordinates=dict(covariance="log(SF)", covariance_sf="linear SF",
            parameter_order="jet, dy, w; C-order flattening within parameter_slices"),
        boundary_parameter_indices=np.flatnonzero((sf - error < 0) & eligible).tolist())
    return covariance_sf


def recompute(directory):
    directory = Path(directory)
    path = directory / "fit_results.json"
    original = path.read_bytes()
    payload = json.loads(original)
    parameters = np.load(directory / "parameters.npy")
    eligible = np.ones(len(parameters), dtype=bool)
    eligible[payload["inapplicable_parameters"]] = False
    total = information_covariance(np.load(directory / "hessian.npy"), eligible)
    data = information_covariance(np.load(directory / "data_hessian.npy"), eligible)
    covariance_sf = update_payload(payload, parameters, total, data)
    payload.setdefault("central_fit_tool_id", json.loads((directory.parent / "run_manifest.json").read_text())["tool_id"])
    payload["uncertainty_recalculation"] = dict(
        original_fit_sha256=hashlib.sha256(original).hexdigest(),
        original_fit_tool_id=json.loads((directory.parent / "run_manifest.json").read_text())["tool_id"],
        algorithm_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        parameters_sha256=hashlib.sha256((directory / "parameters.npy").read_bytes()).hexdigest(),
        central_values_refitted=False)
    error_log = np.sqrt(np.maximum(np.diag(total["covariance"]), 0))
    correlation = np.divide(total["covariance"], error_log[:, None] * error_log[None, :],
        out=np.zeros_like(total["covariance"]), where=(error_log[:, None] * error_log[None, :]) > 0)
    arrays = dict(covariance=total["covariance"], covariance_sf=covariance_sf, correlation=correlation,
        active_parameter_indices=np.flatnonzero(total["active"]),
        data_active_parameter_indices=np.flatnonzero(data["active"]),
        scaled_information_eigenvalues=total["eig"], scaled_information_eigenvectors=total["vectors"],
        data_scaled_information_eigenvalues=data["eig"], data_scaled_information_eigenvectors=data["vectors"])
    with tempfile.TemporaryDirectory(prefix=".covariance_", dir=directory) as tmp:
        tmp = Path(tmp)
        for name, value in arrays.items():
            np.save(tmp / (name + ".npy"), value)
        (tmp / path.name).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        if path.read_bytes() != original:
            raise ValueError("Fit changed during covariance calculation")
        for name in arrays:
            (tmp / (name + ".npy")).replace(directory / (name + ".npy"))
        (tmp / path.name).replace(path)
    return payload


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit-dir", required=True, type=Path)
    a = p.parse_args()
    result = recompute(a.fit_dir)
    print(json.dumps({k: result[k] for k in ("regularized_rank", "errors_valid", "boundary_parameter_indices")}))
