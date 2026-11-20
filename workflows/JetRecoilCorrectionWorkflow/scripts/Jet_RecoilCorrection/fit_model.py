"""Fit reconstructed event templates with fixed prefit normalization and jet-SF priors."""

import hashlib
import os

import json

from pathlib import Path

import time

import numpy as np

from scipy import linalg

from scipy.optimize import lsq_linear

from sf_covariance import information_covariance, update_payload

import config

from sample_groups import write_json

from conditional_model import observation_projection, design, model_and_jacobian

from w_sf import W_SF_DEFINITION, N_W_SF


def edge_metadata(edges):
    return ["inf" if np.isposinf(edge) else float(edge) for edge in edges]


def full_event_metadata(region, is_data):
    payload = config.analysis_metadata(region, is_data)
    fields = {
        key: value for key, value in payload.items() if key != "analysis_signature"
    }
    fields["partition_rule"] = "all"
    return {
        **fields,
        "analysis_signature": hashlib.sha256(
            json.dumps(fields, sort_keys=True).encode()
        ).hexdigest(),
    }


def configure(era, manifest):
    if (
        manifest["model"]["shared_jet_sf_across_eras"]
        != manifest["model"]["shared_dy_sf_across_eras"]
    ):
        raise ValueError("Jet and DY era-sharing modes must agree")
    if (
        manifest["model"]["max_selected_jets"] != config.MAX_JETS
        or manifest["model"]["jet_overflow_policy"]
        != "stable_corrected_pt_descending_take_first_32"
    ):
        raise ValueError("Frozen jet truncation policy differs from this workflow")
    selector = manifest["configuration"]["selection"]["jet_selection"]
    edges = [float(x) for x in manifest["model"]["jet_pt_edges"]]
    config.configure_reference(era, selector, jet_pt_edges=edges)
    config.MC_VETO_CELLS = ()
    selection = config.selection_metadata()
    if (
        selection["jet_pt_edges"] != manifest["model"]["jet_pt_edges"]
        or selection["jet_eta_edges"] != manifest["model"]["jet_eta_edges"]
    ):
        raise ValueError("Jet binning differs from the current configuration")
    if (
        manifest["model"]["dy_pt_edges"] != edge_metadata(config.DY_PT_EDGES)
        or manifest["model"]["dy_njet_categories"] != config.DY_NJET_CATEGORIES
    ):
        raise ValueError("Reco DY-SF binning differs from the frozen configuration")


def compress(keys, values):
    if not len(keys):
        return keys, values
    signature_size = keys.shape[1] * keys.dtype.itemsize
    _, index, inverse = np.unique(
        np.ascontiguousarray(keys).view(f"V{signature_size}").ravel(),
        return_index=True,
        return_inverse=True,
    )
    summed = np.column_stack(
        [
            np.bincount(inverse, weights=values[:, column], minlength=len(index))
            for column in range(6)
        ]
    )
    maximum = np.zeros(len(index))
    np.maximum.at(maximum, inverse, values[:, 6])
    return keys[index], np.column_stack([summed, maximum])


def combine(pieces):
    if not pieces:
        return np.empty((0, 105), np.uint16), np.empty((0, 7))
    return compress(
        np.concatenate([p[0] for p in pieces]), np.concatenate([p[1] for p in pieces])
    )


class SignatureAccumulator:
    """Merge equal-size generations so each input participates in logarithmically many reductions."""

    def __init__(self):
        self.levels = []

    def add(self, piece):
        level = 0
        while level < len(self.levels) and self.levels[level] is not None:
            previous = self.levels[level]
            self.levels[level] = None
            piece = combine([previous, piece])
            del previous
            level += 1
        if level == len(self.levels):
            self.levels.append(piece)
        else:
            self.levels[level] = piece

    def finish(self):
        pieces = [p for p in self.levels if p is not None]
        self.levels = []
        return combine(pieces)


def artifact_root(run, manifest, name):
    return Path(manifest.get(f"shared_{name}_dir", run))


def signatures(arrays, region, is_dy, *, is_data=False, statistics=None):
    """Vectorize the existing observation_signature convention; W observations are local."""
    import awkward as ak
    from until import selected_jets, dilepton_kinematics, bin_indices

    jets = selected_jets(arrays, is_data, statistics=statistics)
    cells = jets[4]
    njet = ak.to_numpy(ak.num(cells))
    n = len(arrays)
    keys = np.full((n, 105), 65535, np.uint16)
    keys[:, 0] = njet + int(region in config.Z_REGIONS)
    keys[:, 1] = njet
    keys[:, 3] = int(is_dy)
    kinematic = config.N_PT * config.N_ETA
    obs = cells % kinematic
    if region in config.Z_REGIONS:
        pt, phi, _ = dilepton_kinematics(arrays, region)
        ptbin, valid = bin_indices(pt, config.DY_PT_EDGES)
        if not np.all(valid):
            raise ValueError("Dilepton pT outside observation bins")
        jetphi = ak.to_numpy(ak.flatten(jets[2]))
        delta = jetphi - np.repeat(phi, njet)
        recoil = (
            np.searchsorted(
                config.RECOIL_DPHI_EDGES[1:-1],
                np.abs(np.arctan2(np.sin(delta), np.cos(delta))),
                side="right",
            )
            + 1
        )
        recoil[np.repeat(pt, njet) < config.RECOIL_PT_THRESHOLDS[region]] = 0
        obs = obs + ak.unflatten(recoil * kinematic, njet)
        keys[:, 2] = ptbin * config.DY_NJET_CATEGORIES + np.minimum(
            njet, config.DY_NJET_CATEGORIES - 1
        )
    if region == "wcr":
        phi = np.asarray(arrays["single_muon_phi"])
        delta = ak.to_numpy(ak.flatten(jets[2])) - np.repeat(phi, njet)
        recoil = (
            np.searchsorted(
                config.RECOIL_DPHI_EDGES[1:-1],
                abs(np.arctan2(np.sin(delta), np.cos(delta))),
                side="right",
            )
            + 1
        )
        recoil[
            np.repeat(np.asarray(arrays["single_muon_pt"]), njet)
            < config.RECOIL_PT_THRESHOLDS["wcr"]
        ] = 0
        obs = obs + ak.unflatten(recoil * kinematic, njet)
    keys[:, 4:37] = ak.to_numpy(ak.fill_none(ak.pad_none(obs, 33, clip=True), 65535))
    if region in config.Z_REGIONS:
        keys[np.arange(n), 4 + njet] = config.DY_OBS_OFFSET + keys[:, 2]
    keys[:, 4:37].sort(axis=1)
    keys[:, 37:69] = ak.to_numpy(ak.fill_none(ak.pad_none(cells, 32, clip=True), 65535))
    keys[:, 37:69].sort(axis=1)
    return keys


def check_inputs(record):
    for item in record["inputs"]:
        stat = Path(item["path"]).stat()
        if stat.st_size != item["size"] or stat.st_mtime_ns != item["mtime_ns"]:
            raise ValueError(f"Input changed: {item['path']}")


def fit_directory(run, manifest, era=None):
    if manifest["model"]["shared_jet_sf_across_eras"] or len(manifest["eras"]) == 1:
        return run / "fit"
    if era not in manifest["eras"]:
        raise ValueError(f"Independent fit requires a concrete era: {era}")
    return run / "fit" / era


def load_prepared(run, manifest, record):
    path = (
        artifact_root(run, manifest, "cache")
        / "cache"
        / record["era"]
        / record["region"]
        / f"{record['sample']}__all.npz"
    )
    meta = json.loads(path.with_suffix(".json").read_text())
    identity = meta["model"] == manifest["model"]
    if (
        not identity
        or meta["partition"] != "all"
        or meta["max_events"] != manifest["max_events"]
        or meta["inputs"] != record["inputs"]
    ):
        raise ValueError(f"Prepared cache layout/partition/input mismatch: {path}")
    if (
        meta.get("config_sha256") != manifest["configuration"]["config_sha256"]
        or meta.get("selection_signature")
        != manifest["configuration"]["selection"]["selection_signature"]
    ):
        raise ValueError(f"Prepared cache configuration mismatch: {path}")
    if (
        meta.get("partition_rule") != "all"
        or meta.get("analysis_signature")
        != full_event_metadata(record["region"], record["sample"] == "data")[
            "analysis_signature"
        ]
    ):
        raise ValueError(f"Prepared cache analysis mismatch: {path}")
    if record["region"] == "wcr" and meta.get("mumet_schema") != 2:
        raise ValueError(f"Missing fine muon+MET cache schema: {path}")
    with np.load(path) as cache:
        return cache["keys"], cache["statistics"]


def muon_observation_bins(model):
    edges = model.get("muon_pt_edges")
    return len(edges) - 1 if edges is not None else 0


def load_context(run, manifest, era, region, alpha=1.0):
    configure(era, manifest)
    observations = config.N_ZCR_OBS if region in config.Z_REGIONS else config.N_WCR_OBS
    keys, values, families = [], [], []
    samples = [
        r
        for r in manifest["samples"]
        if (r["era"], r["region"]) == (era, region) and r["sample"] != "data"
    ]
    names = sorted({r["process"] for r in samples})
    for record in samples:
        key, value = load_prepared(run, manifest, record)
        value[:, [0, 2, 6]] *= alpha
        value[:, 1] *= alpha**2
        keys.append(key)
        values.append(value)
        families.append(np.full(len(key), names.index(record["process"]), dtype=int))
    return design(
        np.concatenate(keys),
        np.concatenate(values),
        np.concatenate(families),
        config.N_JET_CELLS,
        observations,
        region,
        names,
        config.N_DY_CELLS,
        use_w_sf=manifest["model"].get("w_sf_mode") == "reco_njet_mumet_pt",
        muon_bins=muon_observation_bins(manifest["model"]),
    )


def weighted_summary(values, weights):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    if not len(values):
        return dict(count=0)
    absolute = np.abs(weights)
    order = np.argsort(values)
    cumulative = np.cumsum(absolute[order])
    quantiles = (
        [
            values[order[np.searchsorted(cumulative, q * cumulative[-1], side="left")]]
            for q in (0.16, 0.5, 0.84)
        ]
        if cumulative[-1] > 0
        else [None] * 3
    )
    return dict(
        min=float(values.min()),
        median=float(quantiles[1]),
        q16=float(quantiles[0]),
        q84=float(quantiles[2]),
        max=float(values.max()),
        signed_weighted_mean=float(np.sum(values * weights) / np.sum(weights))
        if np.sum(weights) != 0
        else None,
    )


def json_finite_array(values):
    values = np.asarray(values, dtype=float)
    return np.where(np.isfinite(values), values, None).tolist()


def load_data(run, manifest, era, region):
    record = next(
        r
        for r in manifest["samples"]
        if (r["era"], r["region"], r["sample"]) == (era, region, "data")
    )
    keys, values = load_prepared(run, manifest, record)
    h = observation_projection(
        keys,
        config.N_ZCR_OBS if region in config.Z_REGIONS else config.N_WCR_OBS,
        region,
        manifest["model"].get("w_sf_mode") == "reco_njet_mumet_pt",
        muon_observation_bins(manifest["model"]),
    )
    return dict(
        h=h,
        sumw=values[:, 0],
        sumw2=values[:, 1],
        observed=np.asarray(h.T @ values[:, 0]).ravel(),
        njet=keys[:, 1].astype(int),
        path=str(run / "cache" / era / region / "data__all.npz"),
    )


def inapplicable_candidates(manifest):
    """Find jet-SF cells covered by the configured eta and pT veto."""
    selector = manifest["configuration"]["selection"]["jet_selection"]
    if selector not in ("he50", "jme"):
        return np.empty(0, dtype=int)
    pt = np.asarray([float(x) for x in manifest["model"]["jet_pt_edges"]])
    eta = np.asarray(manifest["model"]["jet_eta_edges"], dtype=float)
    eta_low = np.minimum(np.abs(eta[:-1]), np.abs(eta[1:]))
    eta_high = np.maximum(np.abs(eta[:-1]), np.abs(eta[1:]))
    veto = (pt[1:, None] <= 50) & (eta_low[None, :] >= 2.5)
    if selector == "he50":
        veto &= eta_high[None, :] <= 3.0
    return np.flatnonzero(np.tile(veto.reshape(-1), 2))


def sf_prior_indices(name, specification, slices, eligible):
    """Select eligible SF parameters covered by a configured Gaussian prior."""
    start, stop = slices[name]
    selected = np.arange(start, stop)[eligible[start:stop]]
    if "abs_eta_max" in specification:
        limit = float(specification["abs_eta_max"])
        if (
            name != "jet_matched"
            or not np.isfinite(limit)
            or limit <= 0
            or stop - start != config.N_PT * config.N_ETA
        ):
            raise ValueError(f"Invalid eta-restricted SF prior: {name}")
        eta = np.asarray(config.ETA_EDGES, dtype=float)
        eta_low = np.minimum(np.abs(eta[:-1]), np.abs(eta[1:]))
        eta_high = np.maximum(np.abs(eta[:-1]), np.abs(eta[1:]))
        inside = (eta_low < limit) & (eta_high <= limit)
        selected = selected[inside[(selected - start) % config.N_ETA]]
    return selected


def sf_prior_rows(parameters, priors, slices, eligible):
    """Gaussian constraints on positive SFs, differentiated in log coordinates."""
    indices, widths, centers = [], [], []
    for name, specification in priors.items():
        selected = sf_prior_indices(name, specification, slices, eligible)
        if (
            specification["distribution"] != "gaussian"
            or not np.isfinite([specification["sigma"], specification["center"]]).all()
            or specification["sigma"] <= 0
        ):
            raise ValueError(f"Invalid SF prior: {name}")
        indices.extend(selected)
        widths.extend([specification["sigma"]] * len(selected))
        centers.extend([specification["center"]] * len(selected))
    indices = np.asarray(indices, dtype=int)
    sf = np.exp(parameters[indices])
    widths = np.asarray(widths)
    residual = (sf - np.asarray(centers)) / widths
    jacobian = np.zeros((len(indices), len(parameters)))
    jacobian[np.arange(len(indices)), indices] = sf / widths
    return residual, jacobian


def bounded_log_sf_step(information, gradient, scale, damping):
    """Solve the damped quadratic with an independent unit log-SF step box."""
    factor = linalg.cholesky(information + damping * np.eye(len(scale)), lower=False)
    step = linalg.cho_solve((factor, False), -gradient)
    if np.all(np.abs(step) <= scale):
        return step / scale
    target = linalg.solve_triangular(factor.T, -gradient, lower=True)
    result = lsq_linear(
        factor,
        target,
        bounds=(-scale, scale),
        method="bvls",
        tol=1e-10,
        max_iter=2 * len(scale),
    )
    if not result.success:
        raise RuntimeError(f"Bounded log-SF step failed: {result.message}")
    return result.x / scale


def fit_regions(model):
    regions = model["fit_regions"]
    if regions not in (["zcr", "wcr"], ["zee", "wcr"]):
        raise ValueError(f"Invalid fit regions: {regions}")
    return regions


def fit(args, manifest):
    configure(args.era, manifest)
    if manifest.get("stable_summation", False):
        os.environ["RECOIL_STABLE_SUMMATION"] = "1"
    out = fit_directory(args.run_dir, manifest, args.era)
    out.mkdir(parents=True, exist_ok=True)
    contexts, preflight, normalization_by_era = {}, {}, {}
    k = config.N_JET_CELLS
    dy_stop = k + config.N_DY_CELLS
    use_w_sf = manifest["model"].get("w_sf_mode") == "reco_njet_mumet_pt"
    w_stop = dy_stop + (N_W_SF if use_w_sf else 0)
    start = np.zeros(w_stop)
    fit_eras = (
        manifest["eras"]
        if manifest["model"]["shared_jet_sf_across_eras"]
        else [args.era]
    )
    for era in fit_eras:
        for region in fit_regions(manifest["model"]):
            key = f"{era}/{region}"
            mc = load_context(args.run_dir, manifest, era, region)
            data = load_data(args.run_dir, manifest, era, region)
            alpha = float(data["sumw"].sum() / mc["sumw"].sum())
            if not np.isfinite(alpha) or alpha <= 0:
                raise ValueError(
                    f"Invalid prefit Data/MC normalization: {key}: {alpha}"
                )
            mc = load_context(args.run_dir, manifest, era, region, alpha)
            normalization_by_era.setdefault(era, {})[region] = alpha
            cov = (
                (data["h"].T @ data["h"].multiply(data["sumw2"][:, None]))
                + (mc["h"].T @ mc["h"].multiply(mc["sums"][:, 1, None]))
            ).toarray()
            eig, vec = linalg.eigh((cov + cov.T) * 0.5)
            keep = eig > max(eig[-1] * 1e-12, 1e-12)
            whiten = vec[:, keep].T / np.sqrt(eig[keep])[:, None]
            np.save(out / f"{era}_{region}_observation_covariance.npy", cov)
            contexts[key] = dict(mc=mc, data=data, whiten=whiten, covariance=cov)
            neff = np.divide(
                mc["nominal"] ** 2,
                mc["nominal_square"],
                out=np.zeros_like(mc["nominal"]),
                where=mc["nominal_square"] > 0,
            )
            preflight[key] = dict(
                data_events=float(data["sumw"].sum()),
                mc_events=float(mc["sumw"].sum()),
                covariance_rank=int(keep.sum()),
                process_names=mc["group_names"],
                process_effective_statistics=neff.tolist(),
                low_statistics_processes=[
                    name
                    for name, n in zip(mc["group_names"], neff)
                    if n < manifest["model"]["minimum_effective_mc_statistics"]
                ],
            )
            print(
                f"LOADED {key}: signatures={len(mc['sumw'])} covariance_rank={keep.sum()}",
                flush=True,
            )
    eligible = np.ones(len(start), dtype=bool)
    eligible[inapplicable_candidates(manifest)] = False
    tied, masters = np.empty(0, int), np.empty(0, int)
    fit_priors = manifest.get("fit_priors", {})
    slices = dict(
        jet=[0, k],
        jet_matched=[0, k // 2],
        jet_unmatched=[k // 2, k],
        dy=[k, dy_stop],
        w=[dy_stop, w_stop],
    )
    if "jet" in fit_priors and ({"jet_matched", "jet_unmatched"} & fit_priors.keys()):
        raise ValueError("Overlapping jet and matched-jet SF priors")

    def evaluate(parameters):
        residual, jacobians = [], []
        for context in contexts.values():
            mu, jac = model_and_jacobian(parameters, context["mc"])
            residual.append(context["whiten"] @ (context["data"]["observed"] - mu))
            jacobians.append(-context["whiten"] @ jac)
        return (
            np.concatenate(residual),
            np.vstack(jacobians),
            *sf_prior_rows(parameters, fit_priors, slices, eligible),
        )

    def combine_rows(parts):
        return np.concatenate(parts[::2]), np.vstack(parts[1::2])

    x = start.copy()
    if manifest.get("initial_parameters"):
        x = np.load(manifest["initial_parameters"]).copy()
        if x.shape != start.shape or not np.isfinite(x).all():
            raise ValueError("Invalid initial log-SF checkpoint")
    parts = evaluate(x)
    residual, jacobian = combine_rows(parts)
    initial_chi2 = float(parts[0] @ parts[0])
    history, rejected_examples = [], []
    rejected, damping, success = 0, 1e-3, False
    began = time.monotonic()
    for iteration in range(manifest["model"]["max_iterations"]):
        gradient = jacobian.T @ residual
        hessian = jacobian.T @ jacobian
        scale = np.sqrt(np.maximum(np.diag(hessian), 1e-20))
        active = (np.diag(hessian) > 1e-16) & eligible
        scaled_gradient = gradient[active] / scale[active]
        record = dict(
            iteration=iteration,
            reco_chi2=float(parts[0] @ parts[0]),
            sf_prior_chi2=float(parts[2] @ parts[2]),
            objective=float(residual @ residual),
            scaled_gradient_max=float(abs(scaled_gradient).max(initial=0)),
            damping=damping,
            rejected_invalid_proposals=rejected,
            elapsed_seconds=time.monotonic() - began,
        )
        history.append(record)
        print("FIT " + json.dumps(record), flush=True)
        write_json(out / "iteration_status.json", record)
        np.save(out / "parameters_checkpoint.npy", x)
        if record["scaled_gradient_max"] < 1e-5:
            success = True
            break
        if (manifest.get("refine_stationarity", False) and len(history) >= 4
                and history[-4]["scaled_gradient_max"] - record["scaled_gradient_max"] < 1e-8
                and history[-4]["objective"] - record["objective"] < 1e-7):
            print("REFINE_STALLED_STATIONARITY", flush=True)
            break
        information = (
            hessian[np.ix_(active, active)] / scale[active, None] / scale[None, active]
        )
        accepted = False
        for attempt in range(16):
            step = bounded_log_sf_step(
                information, scaled_gradient, scale[active], damping
            )
            proposal = x.copy()
            proposal[active] += step
            try:
                trial_parts = evaluate(proposal)
                trial_residual, trial_jacobian = combine_rows(trial_parts)
                finite = np.isfinite(trial_residual).all()
            except FloatingPointError as error:
                finite = False
                rejected += 1
                if len(rejected_examples) < 10:
                    rejected_examples.append(str(error))
            # Avoid subtracting two large, nearly equal squared objectives.
            decrease = (np.sum((trial_residual.astype(np.longdouble) - residual)
                               * (trial_residual.astype(np.longdouble) + residual))
                        if finite else np.inf)
            if finite and decrease < 0:
                x, residual, jacobian, parts = (
                    proposal,
                    trial_residual,
                    trial_jacobian,
                    trial_parts,
                )
                damping = max(damping * 0.3, 1e-12)
                accepted = True
                break
            damping *= 10
        if not accepted:
            break
    if not success and manifest.get("refine_stationarity", False):
        from scipy.optimize import least_squares
        origin = x.copy()
        scale_refine = np.sqrt(np.maximum(np.sum(jacobian * jacobian, axis=0), 1e-20))
        last = [None, None]

        def refinement_evaluate(z):
            if last[0] is None or not np.array_equal(z, last[0]):
                last[:] = [z.copy(), evaluate(origin + z / scale_refine)]
            return combine_rows(last[1])

        refinement = least_squares(
            lambda z: refinement_evaluate(z)[0], np.zeros_like(x),
            jac=lambda z: refinement_evaluate(z)[1] / scale_refine,
            method="trf", ftol=None, xtol=1e-13, gtol=1e-7,
            max_nfev=100, verbose=2,
        )
        candidate = origin + refinement.x / scale_refine
        refined_parts = evaluate(candidate)
        refined_residual, refined_jacobian = combine_rows(refined_parts)
        objective_delta = np.sum((refined_residual.astype(np.longdouble) - residual)
                                 * (refined_residual.astype(np.longdouble) + residual))
        if objective_delta <= 1e-10:
            x, residual, jacobian, parts = candidate, refined_residual, refined_jacobian, refined_parts
        grad_scale = np.sqrt(np.maximum(np.sum(jacobian * jacobian, axis=0), 1e-20))
        gradient_max = float(np.max(np.abs((jacobian.T @ residual)[eligible] / grad_scale[eligible])))
        success = gradient_max < 1e-5
        record = dict(iteration=len(history), stage="trust_region_refinement",
                      objective=float(residual @ residual), scaled_gradient_max=gradient_max,
                      optimizer_message=str(refinement.message), converged=success)
        history.append(record)
        write_json(out / "iteration_status.json", record)
        print("REFINEMENT " + json.dumps(record), flush=True)
    data_hessian = parts[1].T @ parts[1]
    hessian = jacobian.T @ jacobian
    total_info = information_covariance(hessian, eligible)
    data_info = information_covariance(data_hessian, eligible)
    covariance = total_info["covariance"]
    active, eig, vectors, measured = (total_info[key] for key in ("active", "eig", "vectors", "measured"))
    data_active, data_eig, data_vectors, data_measured = (data_info[key] for key in ("active", "eig", "vectors", "measured"))
    identifiable_parameter = total_info["identifiable"]
    error = np.sqrt(np.maximum(np.diag(covariance), 0))
    error[~identifiable_parameter] = np.nan
    if len(tied):
        covariance[tied, :] = covariance[masters, :]
        covariance[:, tied] = covariance[:, masters]
        error[tied] = error[masters]
        identifiable_parameter[tied] = identifiable_parameter[masters]
    correlation = np.divide(
        covariance,
        error[:, None] * error[None, :],
        out=np.zeros_like(covariance),
        where=error[:, None] * error[None, :] > 0,
    )
    np.save(out / "active_parameter_indices.npy", np.flatnonzero(active))
    np.save(out / "data_active_parameter_indices.npy", np.flatnonzero(data_active))
    np.save(out / "sf_prior_hessian.npy", parts[3].T @ parts[3])
    for name, array in dict(
        parameters=x,
        hessian=hessian,
        data_hessian=data_hessian,
        covariance=covariance,
        correlation=correlation,
        scaled_information_eigenvalues=eig,
        scaled_information_eigenvectors=vectors,
        data_scaled_information_eigenvalues=data_eig,
        data_scaled_information_eigenvectors=data_vectors,
    ).items():
        np.save(out / f"{name}.npy", array)
    predictions, diagnostics, chi2_by_context = {}, {}, {}
    for key, context in contexts.items():
        mc, data = context["mc"], context["data"]
        mu, _, details = model_and_jacobian(x, mc, return_details=True)
        nominal = np.asarray(mc["h"].T @ mc["sumw"]).ravel()
        r = context["whiten"] @ (data["observed"] - mu)
        chi2_by_context[key] = float(r @ r)
        diagnostics[key] = dict(
            process_names=mc["group_names"],
            nominal_sumw=mc["nominal"].tolist(),
            nominal_sumw2=mc["nominal_square"].tolist(),
            corrected_sumw=np.asarray(mc["g"] @ details["weights"]).ravel().tolist(),
            maximum_event_sf=float(
                (details["raw"] * details["dy"] * details["w"]).max(initial=1)
            ),
        )
        predictions[key] = dict(
            data=data["observed"].tolist(),
            nominal=nominal.tolist(),
            corrected=mu.tolist(),
        )
        if mc["region"] == "wcr" and use_w_sf:
            start_muon = config.N_WCR_OBS + 4
            stop_muon = start_muon + muon_observation_bins(manifest["model"])
            predictions[key]["observation_blocks"] = dict(
                jet_pt_eta=[0, config.N_WCR_OBS],
                reco_njet=[config.N_WCR_OBS, start_muon],
                muon_pt=[start_muon, stop_muon],
                mumet_pt_njet=[stop_muon, stop_muon + N_W_SF],
            )
    regularized_rank, rank = int(measured.sum()), int(data_measured.sum())
    reco_chi2, sf_prior_chi2 = float(parts[0] @ parts[0]), float(parts[2] @ parts[2])
    prior_chi2 = {}
    for name, specification in fit_priors.items():
        indices = sf_prior_indices(name, specification, slices, eligible)
        pulls = (np.exp(x[indices]) - specification["center"]) / specification["sigma"]
        prior_chi2[name] = float(pulls @ pulls)
    parameter_count = int(eligible.sum())
    valid = bool(
        success
        and regularized_rank == parameter_count
        and identifiable_parameter[eligible].all()
    )
    payload = dict(
        model=manifest["model"],
        status="exploratory_unpublished",
        eras=fit_eras,
        fit_regions=fit_regions(manifest["model"]),
        validation_regions=manifest["model"]["validation_regions"],
        partition="all",
        optimizer_step_method="bounded_log_sf_gauss_newton",
        maximum_log_sf_step=1.0,
        normalization_by_era=normalization_by_era,
        fit_priors=fit_priors,
        sf_prior_chi2=sf_prior_chi2,
        sf_prior_chi2_by_type=prior_chi2,
        sf_constraint_rows=len(parts[2]),
        optimizer_converged=success,
        fit_success=valid,
        errors_valid=valid,
        regularized_errors_valid=valid,
        identifiable=rank == parameter_count,
        data_identifiable=rank == parameter_count,
        covariance_method="Pseudoinverse of Gaussian-prior-regularized Gauss-Newton information in log-SF coordinates",
        chi2=reco_chi2,
        initial_chi2=initial_chi2,
        chi2_by_context=chi2_by_context,
        reco_chi2=reco_chi2,
        objective=reco_chi2 + sf_prior_chi2,
        covariance_rank=sum(c["whiten"].shape[0] for c in contexts.values()),
        parameter_count=parameter_count,
        grid_parameter_count=len(x),
        active_parameters=int(active.sum()),
        rank=rank,
        data_rank=rank,
        regularized_rank=regularized_rank,
        unsupported_parameters=np.flatnonzero(~active & eligible).tolist(),
        inapplicable_parameters=np.flatnonzero(~eligible).tolist(),
        parameter_slices=slices,
        jet_sf_applicable=eligible[:k].reshape(2, config.N_PT, config.N_ETA).tolist(),
        jet_selection=manifest["configuration"]["selection"]["jet_selection"],
        jet_sf=np.exp(x[:k]).reshape(2, config.N_PT, config.N_ETA).tolist(),
        jet_sf_error=json_finite_array(
            (np.exp(x[:k]) * error[:k]).reshape(2, config.N_PT, config.N_ETA)
        ),
        jet_sf_error_valid=identifiable_parameter[:k]
        .reshape(2, config.N_PT, config.N_ETA)
        .tolist(),
        dy_sf=np.exp(x[k:dy_stop])
        .reshape(config.N_DY_PT, config.DY_NJET_CATEGORIES)
        .tolist(),
        dy_sf_error=json_finite_array(
            (np.exp(x[k:dy_stop]) * error[k:dy_stop]).reshape(
                config.N_DY_PT, config.DY_NJET_CATEGORIES
            )
        ),
        dy_sf_error_valid=identifiable_parameter[k:dy_stop]
        .reshape(config.N_DY_PT, config.DY_NJET_CATEGORIES)
        .tolist(),
        dy_normalization_mode="free",
        dy_parameterization="positive_log",
        dy_pt_edges=edge_metadata(config.DY_PT_EDGES),
        dy_njet_categories=config.DY_NJET_CATEGORIES,
        dy_sf_applied_regions=manifest["model"]["dy_sf_applied_regions"],
        dy_sf_excluded_processes=["DYTau"],
        weak_directions=int((~measured).sum()),
        data_weak_directions=int((~data_measured).sum()),
        total_information_eigenvalues=eig.tolist(),
        data_information_eigenvalues=data_eig.tolist(),
        rejected_invalid_proposals=rejected,
        rejected_proposal_examples=rejected_examples,
        history=history,
        preflight=preflight,
        process_diagnostics=diagnostics,
        predictions=predictions,
        limitations=[
            "Low-statistics bins retained with unchanged binning",
            "Frozen nominal event-level GLS covariance; SF uncertainty not propagated to validation",
            "Cross-channel DY transfer is checked in independent ZCR/SRSB validation samples",
        ],
    )
    if use_w_sf:
        payload.update(
            w_sf=np.exp(x[dy_stop:w_stop]).tolist(),
            w_sf_error=json_finite_array(
                np.exp(x[dy_stop:w_stop]) * error[dy_stop:w_stop]
            ),
            w_sf_error_valid=identifiable_parameter[dy_stop:w_stop].tolist(),
            w_sf_correlation=correlation[dy_stop:w_stop, dy_stop:w_stop].tolist(),
            w_normalization_mode="free",
            w_parameterization="positive_log",
            w_sf_applied_regions=["wcr"],
        )
    payload.update(
        {
            name: manifest["model"][name]
            for name in (
                "w_sf_mode",
                "w_mumet_pt_edges",
                "w_sf_shape",
                "w_sf_index",
                "w_njet_edges",
                "w_njet_labels",
            )
        }
    )
    payload["central_fit_tool_id"] = os.environ.get("RECOIL_EXECUTION_TOOL_ID", manifest.get("execution_tool_id", manifest["tool_id"]))
    covariance_sf = update_payload(payload, x, total_info, data_info)
    np.save(out / "covariance_sf.npy", covariance_sf)
    write_json(out / "fit_results.json", payload)
    print(
        "RESULT "
        + json.dumps(
            {
                key: payload[key]
                for key in [
                    "reco_chi2",
                    "sf_prior_chi2",
                    "objective",
                    "optimizer_converged",
                    "errors_valid",
                    "data_rank",
                    "regularized_rank",
                    "parameter_count",
                ]
            }
        ),
        flush=True,
    )
