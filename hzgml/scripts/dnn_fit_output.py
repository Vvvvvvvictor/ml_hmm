"""Compact, event-aligned DNN outputs and systematic inference (no TF dependency)."""

import re

import numpy as np
import pandas as pd


FIT_COLUMNS = ["run", "lumi", "event", "source_year", "diMufsr_kit_BSC_mass",
               "score_t", "eventWeight", "dnn_score_t"]


def systematic_suffix(name):
    match = re.fullmatch(r"(.+?)(up|down)", name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Systematic name must end in Up/Down: {name}")
    return match[1].lower().rstrip("_") + "_" + match[2].lower()


def systematic_plan(branches, variables, aliases, weight="eventWeight", *, allow_nominal_only=False,
                    include_migrations=False):
    """Discover variations, requiring complete JER/JES inputs and full weights."""
    branches = set(branches)
    weight = aliases.get(weight, weight)
    names = {b.split("__", 1)[1] for b in branches if b.startswith(weight + "__")}
    for variable in variables:
        prefix = aliases.get(variable, variable) + "__"
        names.update(b[len(prefix):] for b in branches if b.startswith(prefix))
    if not names:
        if allow_nominal_only:
            return {}
        raise ValueError("MC systematics requested, but no systematic branches were found")
    plan = {}
    seen = set()
    for name in sorted(names):
        suffix = systematic_suffix(name)
        if suffix in seen:
            raise ValueError(f"Systematic naming collision: {name} -> {suffix}")
        seen.add(suffix)
        weight_branch = weight + "__" + name
        if weight_branch not in branches:
            raise KeyError(f"Missing systematic event weight: {weight_branch}")
        replacements = {}
        for variable in variables:
            varied = aliases.get(variable, variable) + "__" + name
            jet_feature = (variable == "njets" or variable.startswith(("jet", "dijet"))
                           or variable in {"z_star", "R_pT", "delta_eta_jj", "delta_phi_jj"}
                           or variable.startswith(("min_delta_phi_dimu_jets", "min_delta_eta_dimu_jets")))
            if name.lower().startswith(("jer", "jes")) and jet_feature and varied not in branches:
                raise KeyError(f"Missing systematic feature: {varied}")
            if varied in branches:
                replacements[variable] = varied
        plan[name] = {"suffix": suffix, "weight": weight_branch, "features": replacements}
        if include_migrations and name.lower().startswith(("jer", "jes")):
            category_branch = aliases.get("cate_index", "cate_index") + "__" + name
            if category_branch not in branches:
                raise KeyError(f"Missing systematic category: {category_branch}")
            plan[name]["category"] = category_branch
    return plan


def prepare_features(handler, data):
    index = data.index
    data = handler._apply_2223_jet_patch(data)
    data = handler._apply_ptdecor(data)
    if handler._FixSBH125:
        handler._apply_fix_mass(data, data.copy())
    if not data.index.equals(index):
        raise ValueError("DNN preprocessing dropped events from the fixed nominal selection")
    values = data[handler.train_variables].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite DNN inputs in the fixed nominal selection")
    return data


def score_test_folds(handler, data):
    folds = (data[handler.randomIndex] % 314159 % 4).to_numpy(dtype=int)
    # WeightedQuantileTransformer returns float64, as in the legacy apply output.
    scores = np.empty(len(data), dtype=np.float64)
    for fold in range(4):
        mask = folds == fold
        if not mask.any():
            continue
        inputs = handler._transform_features(data.loc[mask, handler.train_variables], fold)
        raw_scores = handler._predict_scores(inputs, fold)
        transformed = np.asarray(handler.m_tsfs[fold].transform(raw_scores.reshape(-1, 1))).reshape(-1)
        if not np.isfinite(transformed).all():
            raise ValueError(f"Non-finite transformed DNN score in fold {fold}")
        scores[mask] = transformed
    return scores


def select_jet_migrations(handler, canonical, raw, plan):
    nominal = raw.index.isin(handler.preselect(canonical).index)
    selected = nominal.copy()
    for var in plan.values():
        if "category" in var:
            shifted = canonical.assign(cate_index=raw[var["category"]])
            selected |= raw.index.isin(handler.preselect(shifted).index)
    return selected, nominal[selected]


def fit_output(handler, raw, mapping, plan, category, *, nominal_mask=None):
    """Score selected events, optionally masking each weight by its own category."""
    inputs = raw[list(mapping)].rename(columns=mapping)
    canonical = handler._inject_source_year(inputs.copy())
    result = canonical[["run", "lumi", "event", "source_year"]].copy()
    handler._write_canonical_mass(canonical, result)
    result["score_t"] = score_test_folds(handler, prepare_features(handler, canonical.copy()))
    result["dnn_score_t"] = result["score_t"]
    scale = handler._event_weight_scale if category != "data" else 1.0
    weight = canonical[handler.weight].to_numpy(dtype=np.float64) * scale
    result["eventWeight"] = weight if nominal_mask is None else np.where(nominal_mask, weight, 0.0)
    for variation in plan.values():
        suffix = variation["suffix"]
        if variation["features"]:
            shifted = inputs.copy()
            for variable, branch in variation["features"].items():
                shifted[variable] = raw[branch]
            shifted = prepare_features(handler, handler._inject_source_year(shifted))
            result["score_t_" + suffix] = score_test_folds(handler, shifted)
        else:
            result["score_t_" + suffix] = result["score_t"]
        weight = raw[variation["weight"]].to_numpy(dtype=np.float64) * scale
        if nominal_mask is not None:
            selected = raw[variation["category"]].to_numpy() == 4 if "category" in variation else nominal_mask
            weight = np.where(selected, weight, 0.0)
        result["eventWeight_" + suffix] = weight
    weight_columns = [c for c in result if c.startswith("eventWeight")]
    if not np.isfinite(result[weight_columns].to_numpy()).all():
        raise ValueError("Non-finite event weights in DNN output")
    return result[FIT_COLUMNS + [c for c in result if c not in FIT_COLUMNS]]


def mapping_record(plan):
    return {name: {**variation, "score_output": "score_t_" + variation["suffix"],
                   "weight_output": "eventWeight_" + variation["suffix"]}
            for name, variation in plan.items()}
