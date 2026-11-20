"""Sparse reconstructed-event SF prediction and analytic Jacobian."""

import numpy as np
import os
from scipy import sparse
from w_sf import w_bins, N_W_SF, W_MUMET_COLUMN, w_sf_indices


def contribution(entries, columns, offset=0):
    rows, slots = np.nonzero(entries != 65535)
    return sparse.csr_matrix(
        (np.ones(len(rows)), (rows, entries[rows, slots].astype(int) - offset)),
        shape=(len(entries), columns),
    )


def observation_projection(keys, columns, region, use_w_sf=False, muon_bins=0):
    h = contribution(keys[:, 4:37], columns)
    if use_w_sf and region == "wcr":
        counts = sparse.csr_matrix(
            (np.ones(len(keys)), (np.arange(len(keys)), w_bins(keys[:, 1]))),
            shape=(len(keys), 4),
        )
        h = sparse.hstack([h, counts], format="csr")
    if region == "wcr" and muon_bins:
        indices = keys[:, 112].astype(int)
        if np.any(indices >= muon_bins):
            raise ValueError("Missing muon pT observation")
        muon = sparse.csr_matrix(
            (np.ones(len(keys)), (np.arange(len(keys)), indices)),
            shape=(len(keys), muon_bins),
        )
        h = sparse.hstack([h, muon], format="csr")
    if use_w_sf and region == "wcr":
        index = w_sf_indices(keys[:, 1], keys[:, W_MUMET_COLUMN])
        joint = sparse.csr_matrix(
            (np.ones(len(keys)), (np.arange(len(keys)), index)),
            shape=(len(keys), N_W_SF),
        )
        h = sparse.hstack([h, joint], format="csr")
    return h


def design(
    keys,
    sums,
    families,
    jet_cells,
    observations,
    region,
    group_names,
    dy_cells,
    use_w_sf=False,
    muon_bins=0,
):
    h = observation_projection(keys, observations, region, use_w_sf, muon_bins)
    p = contribution(keys[:, 37:69], jet_cells)
    group = np.asarray(families, dtype=int)
    g = sparse.csr_matrix(
        (np.ones(len(keys)), (group, np.arange(len(keys)))),
        shape=(len(group_names), len(keys)),
    )
    return dict(
        h=h,
        p=p,
        g=g,
        group=group,
        group_names=group_names,
        group_labels=np.arange(len(group_names))[:, None],
        sums=sums,
        sumw=sums[:, 0],
        is_dy=keys[:, 3].astype(bool),
        dy_index=keys[:, 2].astype(int),
        dy_cells=dy_cells,
        is_w=np.asarray([group_names[i] == "W" for i in group]) & (region == "wcr"),
        w_index=(
            w_sf_indices(keys[:, 1], keys[:, W_MUMET_COLUMN])
            if region == "wcr"
            else np.zeros(len(keys), int)
        ),
        w_sf_enabled=use_w_sf,
        njet=keys[:, 1].astype(int),
        nominal=np.asarray(g @ sums[:, 0]).ravel(),
        nominal_absolute=np.asarray(g @ sums[:, 2]).ravel(),
        nominal_square=np.asarray(g @ sums[:, 1]).ravel(),
        region=region,
    )


def model_and_jacobian(parameters, mc, *, apply_dy=None, return_details=False):
    p, h = mc["p"], mc["h"]
    stable = os.environ.get("RECOIL_STABLE_SUMMATION") == "1"
    if stable:
        # Millions of signed contributions must not make objective differences
        # noisier than the declared stationarity threshold. Keep the physical
        # model unchanged; round only the aggregated predictions/Jacobian.
        if "_stable_p" not in mc:
            mc["_stable_p"] = p.astype(np.longdouble)
            mc["_stable_h"] = h.astype(np.longdouble)
        p, h = mc["_stable_p"], mc["_stable_h"]
        parameters = np.asarray(parameters, dtype=np.longdouble)
    k, dy_cells = p.shape[1], int(mc["dy_cells"])
    use_w = mc.get("w_sf_enabled", False)
    expected = k + dy_cells + (N_W_SF if use_w else 0)
    if len(parameters) != expected or not np.isfinite(parameters).all():
        raise ValueError(f"Expected {expected} finite log-SFs")
    if apply_dy is None:
        apply_dy = mc["region"] != "wcr"
    rows_dy = np.flatnonzero(mc["is_dy"]) if apply_dy else np.empty(0, int)
    rows_w = np.flatnonzero(mc["is_w"]) if use_w else np.empty(0, int)
    with np.errstate(over="raise", invalid="raise"):
        raw = np.exp(np.asarray(p @ parameters[:k]).ravel())
        dy, w = np.ones(len(raw), dtype=raw.dtype), np.ones(len(raw), dtype=raw.dtype)
        indices = mc["dy_index"][rows_dy]
        if np.any((indices < 0) | (indices >= dy_cells)):
            raise ValueError("Invalid reconstructed DY-SF index")
        dy[rows_dy] = np.exp(parameters[k + indices])
        if use_w:
            w[rows_w] = np.exp(parameters[k + dy_cells + mc["w_index"][rows_w]])
        weights = mc["sumw"] * raw * dy * w
    prediction = np.asarray(h.T @ weights, dtype=float).ravel()
    dy_design = sparse.csr_matrix(
        (weights[rows_dy], (rows_dy, indices)), shape=(len(raw), dy_cells)
    )
    jacobian = [
        (h.T @ p.multiply(weights[:, None])).toarray(),
        (h.T @ dy_design).toarray(),
    ]
    if use_w:
        w_design = sparse.csr_matrix(
            (weights[rows_w], (rows_w, mc["w_index"][rows_w])), shape=(len(raw), N_W_SF)
        )
        jacobian.append((h.T @ w_design).toarray())
    jacobian = np.asarray(np.column_stack(jacobian), dtype=float)
    if not np.isfinite(prediction).all() or not np.isfinite(jacobian).all():
        raise FloatingPointError("Nonfinite prediction or Jacobian")
    if return_details:
        return prediction, jacobian, {key: np.asarray(value, dtype=float) for key, value in
                                    dict(raw=raw, dy=dy, w=w, weights=weights).items()}
    return prediction, jacobian
