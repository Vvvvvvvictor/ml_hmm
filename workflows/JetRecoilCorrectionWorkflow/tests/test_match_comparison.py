"""Exercise matching, stable ranking and event-correlated percentage errors."""
import sys
from pathlib import Path
import awkward as ak
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/Jet_RecoilCorrection'))
from match_comparison import allocate, accumulate, quantities


def test_match_comparison():
    axes = {'jet_pt': (np.array([0., 50., 100.]), 'pt'), 'jet_eta': (np.array([-5., 0., 5.]), 'eta')}
    # Empty, one jet, tied jets, and a negative-weight event. Cell < 280 is matched.
    pt = ak.Array([[], [40.], [60., 60., 30.], [40., 30.]])
    eta = ak.Array([[], [1.], [-1., 1., -1.], [1., 1.]])
    cells = ak.Array([[], [0], [281, 1, 2], [0, 281]])
    weights = np.array([[1., 1.], [2., 4.], [3., 3.], [-1., -2.]])
    hist = {}
    allocate(hist, axes)
    accumulate(hist, axes, (pt, eta, None, None, cells), weights)
    a = hist['comparison__all_jet_pt__sumw']
    np.testing.assert_allclose(a, [[4, 3], [-1, 3], [5, 3], [-2, 3]])
    leading = hist['comparison__leading_jet_pt__sumw']
    np.testing.assert_allclose(leading, [[1, 0], [0, 3], [2, 0], [0, 3]])
    subleading = hist['comparison__subleading_jet_pt__sumw']
    np.testing.assert_allclose(subleading, [[0, 3], [-1, 0], [0, 3], [-2, 0]])
    cov = hist['comparison__all_jet_pt_covariance']
    # Same negative event contributes both matched and unmatched to the first bin.
    assert cov[0, 0, 1] == 1 and cov[0, 0, 3] == 2
    q = quantities(a, cov, 'matched')
    assert q['valid'].tolist() == [True, True]
    np.testing.assert_allclose(q['change_percent'], [25., 0.])
    q = quantities(a, cov, 'fraction')
    np.testing.assert_allclose(q['before'], [4 / 3, .5])
    assert q['valid'].all()
    same = np.array([[2., 0.], [1., 0.], [2., 0.], [1., 0.]])
    correlated = np.zeros((2, 4, 4)); correlated[0] = np.outer([2., 1., 2., 1.], [2., 1., 2., 1.])
    for part in ('matched', 'unmatched', 'fraction'):
        q = quantities(same, correlated, part)
        assert q['valid'].tolist() == [True, False]
        np.testing.assert_allclose(q['change_percent'][0], 0)
        np.testing.assert_allclose(q['change_error_percent'][0], 0, atol=1e-6)

if __name__ == '__main__':
    test_match_comparison()
    print('matched comparison checks passed')
