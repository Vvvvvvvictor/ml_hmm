"""Region-aware event roles shared by streaming and in-memory DNN preparation.

Region is provenance, never a classifier input. Empty policies retain ggH defaults.
"""
import numpy as np

MASS = "diMufsr_kit_BSC_mass"
REGION = "__input_region__"


def window(mass, limits, upper_inclusive=True):
    mass = np.asarray(mass)
    low, high = limits
    return np.isfinite(mass) & (mass >= low) & ((mass <= high) if upper_inclusive else (mass < high))


def select_input(data, region, policy):
    if not policy:
        return data
    if region not in ("SRSB", "ZCR"):
        raise ValueError(f"Unknown input region: {region}")
    data = data.loc[window(data[MASS], policy[region.lower() + "_window"], region == "SRSB")].copy()
    data[REGION] = region
    return data


def role_masks(data, signal, mc, real_data, policy, mass_window):
    """Select classifier, mass and domain populations without changing weights."""
    mass = data[MASS].to_numpy()
    srsb = (data[REGION].to_numpy() == "SRSB") & window(mass, policy["srsb_window"])
    zcr = (data[REGION].to_numpy() == "ZCR") & window(mass, policy["zcr_window"], False)
    sideband = ((mass >= 110) & (mass < 115)) | ((mass > 135) & (mass <= 150))
    classifier_regions = policy.get("classifier_regions", ["SRSB", "ZCR"])
    if not classifier_regions or set(classifier_regions) - {"SRSB", "ZCR"}:
        raise ValueError(f"Invalid classifier regions: {classifier_regions}")
    score_region = (srsb if "SRSB" in classifier_regions else False) | (zcr if "ZCR" in classifier_regions else False)
    return {
        "score_sig": signal & srsb & score_region,
        "score_bkg": mc & score_region,
        "mass_bkg": mc & srsb & window(mass, mass_window),
        "domain_mc": mc & ((srsb & sideband) | zcr),
        "domain_data": real_data & ((srsb & sideband) | zcr),
    }


def fold_masks(event, fold):
    # Shift the four-way bucket, not the original event number: the latter can
    # overlap train/test at the 314159 wrap boundary.
    bucket = np.asarray(event, dtype=np.int64) % 314159 % 4
    relative = (bucket - fold) % 4
    return {"train": relative >= 2, "val": relative == 1, "test": relative == 0}


def metric_arrays(labels, scores, weights, masses, policy):
    if not policy:
        return labels, scores, weights
    if masses is None:
        raise ValueError("Region-aware AUC requires masses")
    mask = window(masses, policy["auc_window"])
    arrays = tuple(np.asarray(x)[mask] for x in (labels, scores, weights))
    if len(np.unique(arrays[0])) != 2:
        raise ValueError("SRSB AUC requires signal and background")
    return arrays
