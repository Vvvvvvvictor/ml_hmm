"""Exact local fitted-SF covariance of arbitrary bins of the signal skim.

Use accumulate() on chunks, ADD the gradients from all samples sharing a fit,
then call bin_covariance() ONCE. MC statistical sumw2 is a separate quantity.
"""
import numpy as np
import awkward as ak
from scipy import sparse


def accumulate(arrays, bins, n_bins, n_parameters=794):
    bins=np.asarray(bins,int)
    keep=(bins>=0)&(bins<n_bins)&(np.asarray(arrays['cate_index'])>0)
    weight=np.asarray(arrays['eventWeight'],float)[keep]
    indices=arrays['sf_index'][keep]
    counts=arrays['sf_count'][keep]
    length=ak.to_numpy(ak.num(indices))
    derivative=sparse.csr_matrix((ak.to_numpy(ak.flatten(counts))*np.repeat(weight,length),
                                 (np.repeat(bins[keep],length),ak.to_numpy(ak.flatten(indices)))),
                                shape=(n_bins,n_parameters)).toarray()
    return dict(sumw=np.bincount(bins[keep],weights=weight,minlength=n_bins),
                sumw2=np.bincount(bins[keep],weights=weight**2,minlength=n_bins),gradient=derivative)


def bin_covariance(gradient,covariance_log_sf):
    g=np.asarray(gradient,float);c=np.asarray(covariance_log_sf,float)
    if g.shape[1]!=len(c) or not np.isfinite(g).all() or not np.isfinite(c).all():
        raise ValueError('Invalid covariance inputs')
    return g@c@g.T
