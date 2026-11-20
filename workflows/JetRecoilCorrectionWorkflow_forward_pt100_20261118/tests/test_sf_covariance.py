"""Run with the production LCG environment; no ROOT inputs required."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/Jet_RecoilCorrection'))
from sf_covariance import information_covariance, update_payload, propagate_sf_uncertainty

class CovarianceTest(unittest.TestCase):
    def test_coordinate_scaling_and_tiny_sf(self):
        linear = np.array([[4., 1.8], [1.8, 1.]])
        sf = np.array([1e-11, 2.])
        log = sf[:, None] * linear * sf[None, :]
        r = information_covariance(log, [True, True])
        self.assertEqual(r['measured'].sum(), 2)
        c = r['covariance'] * sf[:, None] * sf[None, :]
        np.testing.assert_allclose(c, np.linalg.inv(linear), rtol=1e-13)
        np.testing.assert_allclose(c, c.T)
        self.assertGreater(np.linalg.eigvalsh(c).min(), 0)
        grad = np.array([3., -1.])
        self.assertAlmostEqual(grad @ c @ grad, (grad * sf) @ r['covariance'] @ (grad * sf))
        self.assertNotAlmostEqual(grad @ c @ grad, np.sum(grad**2 * np.diag(c)))
        self.assertAlmostEqual(propagate_sf_uncertainty(grad, c)**2, grad @ c @ grad)

    def test_ordinary_matches_previous_method(self):
        h = np.array([[9., 2.], [2., 4.]])
        np.testing.assert_allclose(information_covariance(h, [True]*2)['covariance'], np.linalg.inv(h))

    def test_true_null_and_zero_curvature(self):
        r = information_covariance(np.ones((2, 2)), [True]*2)
        self.assertEqual(r['measured'].sum(), 1)
        self.assertFalse(r['identifiable'].any())
        r = information_covariance(np.diag([2., 0]), [True]*2)
        np.testing.assert_array_equal(r['identifiable'], [True, False])
        self.assertEqual(r['measured'].sum(), 1)
        r = information_covariance(np.zeros((2,2)), [True]*2)
        self.assertEqual(r['measured'].sum(), 0)
        self.assertFalse(r['identifiable'].any())

    def test_invalid(self):
        for h in (np.array([[np.nan]]), np.array([[np.inf]]), np.array([[-1.]]),
                  np.array([[1., 2.], [2., 1.]]), np.array([[0., 1.], [1., 1.]])):
            with self.assertRaises(ValueError):
                information_covariance(h, [True]*len(h))

    def test_payload_boundary_and_invalidity(self):
        sf = np.exp(np.array([-25., 0.]))
        p = dict(jet_sf=sf.reshape(1,2).tolist(), parameter_slices={'jet':[0,2]},
                 inapplicable_parameters=[], optimizer_converged=True)
        info = information_covariance(np.diag(sf**2), [True]*2)
        c = update_payload(p, np.log(sf), info, info)
        np.testing.assert_allclose(np.diag(c), 1.)
        self.assertTrue(p['errors_valid'])
        self.assertTrue(p['jet_sf_near_boundary'][0][0])
        bad = information_covariance(np.diag([sf[0]**2, 0.]), [True]*2)
        update_payload(p, np.log(sf), bad, bad)
        self.assertFalse(p['errors_valid'])
        self.assertIsNone(p['jet_sf_error'][0][1])

if __name__ == '__main__':
    unittest.main()
