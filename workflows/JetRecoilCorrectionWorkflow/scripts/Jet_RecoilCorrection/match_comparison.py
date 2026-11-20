"""Jet-SF-only comparisons with event-level correlations between stages."""
import json
import awkward as ak
import config
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from histograms import _jet_counts
from fit_statistics import finite

KINDS = ('all', 'leading', 'subleading')
VARIABLES = ('jet_pt', 'jet_eta')
COMPONENTS = ('matched', 'unmatched', 'fraction')


def allocate(hist, axes):
    for kind in KINDS:
        for name in VARIABLES:
            bins = len(axes[name][0]) - 1
            prefix = f'comparison__{kind}_{name}'
            hist[prefix + '__sumw'] = np.zeros((4, bins))
            hist[prefix + '_covariance'] = np.zeros((bins, 4, 4))


def accumulate(hist, axes, jets, weights):
    # Stable sorting matches the selected-jet convention, including ties.
    order = ak.argsort(jets[0], axis=1, ascending=False, stable=True)
    for kind in KINDS:
        chosen = order if kind == 'all' else order[:, :1] if kind == 'leading' else order[:, 1:2]
        matched = jets[4][chosen] < config.N_PT * config.N_ETA
        for name, values in (('jet_pt', jets[0][chosen]), ('jet_eta', jets[1][chosen])):
            edges = axes[name][0]
            matrices = [_jet_counts(values[mask], edges) for mask in (matched, ~matched)]
            n = len(edges) - 1
            prefix = f'comparison__{kind}_{name}'
            # Per bin order: nominal matched/unmatched, jet-SF matched/unmatched.
            for stage in range(2):
                for part in range(2):
                    idx = 2 * stage + part
                    hist[prefix + '__sumw'][idx] += np.asarray(matrices[part].T @ weights[:, stage]).ravel()
                    for other_stage in range(2):
                        for other_part in range(2):
                            j = 2 * other_stage + other_part
                            product = matrices[part].multiply(matrices[other_part])
                            hist[prefix + '_covariance'][:, idx, j] += np.asarray(product.T @ (weights[:, stage] * weights[:, other_stage])).ravel()


def quantities(values, covariance, component):
    """Two stage values and correlated errors, with delta-method percentage."""
    n = values.shape[1]
    x = values.T
    y = np.full((n, 2), np.nan)
    jac = np.zeros((n, 2, 4))
    if component != 'fraction':
        part = 0 if component == 'matched' else 1
        for stage in range(2):
            y[:, stage] = x[:, 2 * stage + part]
            jac[:, stage, 2 * stage + part] = 1
    else:
        for stage in range(2):
            m, u = x[:, 2 * stage], x[:, 2 * stage + 1]
            total = m + u
            ok = total != 0
            y[ok, stage] = m[ok] / total[ok]
            jac[ok, stage, 2 * stage] = u[ok] / total[ok]**2
            jac[ok, stage, 2 * stage + 1] = -m[ok] / total[ok]**2
    cov_y = np.einsum('nsi,nij,ntj->nst', jac, covariance, jac)
    valid = np.isfinite(y).all(axis=1) & (y[:, 0] != 0)
    change = np.full(n, np.nan)
    change[valid] = 100 * (y[valid, 1] / y[valid, 0] - 1)
    grad = np.zeros((n, 2))
    grad[valid, 0] = -100 * y[valid, 1] / y[valid, 0]**2
    grad[valid, 1] = 100 / y[valid, 0]
    variance = np.einsum('ni,nij,nj->n', grad, cov_y, grad)
    error = np.where(valid, np.sqrt(np.maximum(variance, 0)), np.nan)
    errors = np.sqrt(np.maximum(np.diagonal(cov_y, axis1=1, axis2=2), 0))
    errors[~np.isfinite(y)] = np.nan
    return dict(before=y[:, 0], after=y[:, 1], before_error=errors[:, 0], after_error=errors[:, 1], change_percent=change, change_error_percent=error, valid=valid, stage_covariance=cov_y)


def render(era, region, total, out, summary, alpha, axes):
    results = {}
    for kind in KINDS:
        for name in VARIABLES:
            edges, xlabel = axes[name]
            centers = (edges[:-1] + edges[1:]) / 2
            prefix = f'comparison__{kind}_{name}'
            values = total[prefix + '__sumw'] / alpha
            covariance = total[prefix + '_covariance'] / alpha**2
            if kind == 'all':
                bins = len(centers)
                for stage in range(2):
                    expected = total[name + '__split_sumw'][stage].reshape(2, bins) / alpha
                    np.testing.assert_allclose(values[2 * stage:2 * stage + 2], expected, rtol=1e-9, atol=1e-5)
                    split_cov = total[name + '__split_covariance'][stage] / alpha**2
                    for a in range(2):
                        for b in range(2):
                            indices = np.arange(bins)
                            np.testing.assert_allclose(covariance[:, 2 * stage + a, 2 * stage + b], split_cov[a * bins + indices, b * bins + indices], rtol=1e-9, atol=1e-5)
            for component in COMPONENTS:
                result = quantities(values, covariance, component)
                stem = f'{kind}_{name}_{component}_jet_sf_physical'
                fig, (ax, ratio) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, layout='constrained', gridspec_kw={'height_ratios': [3, 1]})
                for stage, color in (('before', '#4c78a8'), ('after', '#e45756')):
                    ax.errorbar(centers, result[stage], yerr=result[stage + '_error'], fmt='.', color=color, label=('Nominal' if stage == 'before' else 'Jet SF only'))
                ratio.errorbar(centers[result['valid']], result['change_percent'][result['valid']], yerr=result['change_error_percent'][result['valid']], fmt='k.')
                ratio.axhline(0, color='gray', linestyle='--')
                ratio.set(xlabel=xlabel, ylabel='Change [%]')
                ax.set(ylabel='Matched fraction' if component == 'fraction' else 'Weighted jets / bin', title=f'{era} {region.upper()} {kind} {component}: nominal → jet SF only\nPhysical yield; MC statistical uncertainty')
                ax.legend()
                path = out / 'match_split' / stem
                path.parent.mkdir(parents=True, exist_ok=True)
                for ext in ('png', 'pdf'): fig.savefig(path.with_suffix('.' + ext), dpi=150)
                summary.savefig(fig)
                plt.close(fig)
                results[stem] = dict(edges=edges, **result)
    (out / 'match_split' / 'jet_sf_comparisons.json').write_text(json.dumps(finite(results), indent=2, allow_nan=False) + '\n')
