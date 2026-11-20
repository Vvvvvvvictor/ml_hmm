import numpy as np

from scipy import linalg


def finite(value):
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def precision(covariance):
    eig, vec = linalg.eigh((covariance + covariance.T) * 0.5)
    keep = eig > max(eig.max(initial=0) * 1e-12, 1e-12)
    return (vec[:, keep] / eig[keep]) @ vec[:, keep].T, int(keep.sum())


def chi2(residual, covariance):
    inv, rank = precision(covariance)
    return dict(chi2=float(residual @ inv @ residual), rank=rank)
