"""Accumulate event-level histogram moments without dense event/bin matrices."""

import numpy as np
from numba import njit
from scipy import sparse


def one_hot_counts(indices, bins, valid=None):
    """CSR with one entry per valid event; invalid values contribute nothing."""
    indices = np.asarray(indices)
    rows = np.arange(len(indices)) if valid is None else np.flatnonzero(valid)
    return sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.int32), (rows, indices[rows])),
        shape=(len(indices), bins),
    )


@njit(nogil=True, cache=False, fastmath=False, parallel=False)
def _moments(indptr, indices, counts, weights, bins, covariance):
    stages = weights.shape[1]
    sumw = np.zeros((stages, bins))
    sumw2 = np.zeros((stages, bins))
    cov = np.zeros((stages, bins, bins)) if covariance else np.zeros((0, 0, 0))
    for row in range(len(indptr) - 1):
        for pos in range(indptr[row], indptr[row + 1]):
            col, count = indices[pos], counts[pos]
            for stage in range(stages):
                weight = weights[row, stage]
                sumw[stage, col] += weight * count
                sumw2[stage, col] += weight * weight * count * count
                if covariance:
                    for other in range(indptr[row], indptr[row + 1]):
                        cov[stage, col, indices[other]] += (
                            weight * weight * count * counts[other]
                        )
    return sumw, sumw2, cov


def event_moments(counts, weights, *, covariance=False):
    """Return sumw, event-level sumw2 and optional full covariance.

    Combine duplicate (event, bin) entries *before* squaring: two jets from
    one event in the same bin contribute (2*w)**2, rather than 2*w**2.
    """
    counts = sparse.csr_matrix(counts, dtype=np.float64)
    counts.sum_duplicates()
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2 or counts.shape[0] != len(weights):
        raise ValueError("Histogram counts and event weights have incompatible shapes")
    return _moments(
        counts.indptr, counts.indices, counts.data, weights, counts.shape[1], covariance
    )
