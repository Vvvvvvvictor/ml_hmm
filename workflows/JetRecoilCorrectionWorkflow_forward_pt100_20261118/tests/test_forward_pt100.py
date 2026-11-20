"""Cross-check strict per-jet boundaries, event products and exported corrections."""
import json
import os
from pathlib import Path
import sys
import tempfile

import awkward as ak
import correctionlib
import numpy as np

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE / 'scripts/Jet_RecoilCorrection'))
_run = Path(tempfile.mkdtemp(prefix='forward_pt100_test_'))
(_run / 'campaign.json').write_text(json.dumps({'eras': ['2024'], 'validation_regions': ['zee', 'wcr', 'zcr', 'srsb']}))
os.environ['RECOIL_RUN_DIR'] = str(_run)
import config
from until import selected_jets
from fit_model import signatures, inapplicable_candidates
from conditional_model import design, model_and_jacobian
from validation_plots import event_factors
from sf_export import build_payload
from sf_covariance import information_covariance, update_payload


def test_products_boundaries_and_export():
    config.configure_reference('2024')
    model = json.loads((CORE / 'model.json').read_text())['model']
    manifest = dict(eras=['2024'], model=model, configuration={'selection': {'jet_selection': 'inclusive'}})
    fixed = np.zeros(config.N_JET_CELLS, bool)
    fixed[inapplicable_candidates(manifest)] = True
    assert config.N_PT == 7 and config.N_JET_CELLS == 560
    assert np.count_nonzero(~fixed) == 336
    jet = np.where(fixed, 1., 1.2)
    pts = [[35., 35., 100., 35., 35.], [], [35., 35.]]
    etas = [[2.5, -2.5, 3., 3., -3.], [], [3., 3.]]
    n = [len(p) for p in pts]
    a = ak.Array(dict(Jet_corrPt=pts, Jet_corrEta=etas,
                      Jet_corrPhi=[[0.]*i for i in n], Jet_corrMass=[[0.]*i for i in n],
                      Jet_isGenMatched=[[1]*i for i in n],
                      zee_pt=[30., 30., 30.], zee_phi=[0., 0., 0.], zee_mass=[91., 91., 91.]))
    jets = selected_jets(a, False)
    shape = (2, config.N_PT, config.N_ETA)
    fit = dict(eras=['2024'], optimizer_converged=True, fit_success=True, errors_valid=True,
               jet_sf=jet.reshape(shape).tolist(), jet_sf_error=np.where(fixed, 0., 0.1).reshape(shape).tolist(),
               jet_sf_error_valid=np.ones(shape, bool).tolist(), jet_sf_near_boundary=np.zeros(shape, bool).tolist(),
               jet_sf_fixed=fixed.reshape(shape).tolist(), jet_sf_applicable=np.ones(shape, bool).tolist(),
               dy_sf=np.full((26, 5), 1.3).tolist(), dy_sf_error=np.full((26, 5), 0.1).tolist(),
               dy_sf_error_valid=np.ones((26, 5), bool).tolist(), dy_sf_near_boundary=np.zeros((26, 5), bool).tolist(),
               dy_sf_applied_regions=['zee', 'zcr', 'srsb'])
    factors = event_factors(jets, dict(sample='DY', group='DY', region='zee', dy_sf_applicable=True), fit, np.full(3,30.))
    np.testing.assert_allclose(factors[:,1], [1.2**2, 1., 1.2**2])
    np.testing.assert_allclose(factors[:,3], factors[:,1]*1.3)
    keys = signatures(a, 'zee', True)
    assert [np.sum(row != 65535) for row in keys[:,37:69]] == [2, 0, 2]
    sums = np.zeros((3,7)); sums[:,0] = [-2., 3., 4.]
    mc = design(keys, sums, np.zeros(3,int), config.N_JET_CELLS, config.N_ZCR_OBS, 'zee', ['DY'], 130)
    x = np.r_[np.log(jet), np.full(130, np.log(1.3))]
    pred, jac, details = model_and_jacobian(x, mc, return_details=True)
    np.testing.assert_allclose(details['raw'], factors[:,1])
    np.testing.assert_allclose(details['weights'], sums[:,0]*factors[:,3])
    i = int(keys[0,37]); delta=np.zeros_like(x); delta[i]=1e-5
    numerical=(model_and_jacobian(x+delta,mc)[0]-model_and_jacobian(x-delta,mc)[0])/2e-5
    np.testing.assert_allclose(numerical,jac[:,i],rtol=1e-8,atol=1e-8)
    cset = correctionlib.CorrectionSet.from_string(json.dumps(build_payload(manifest, fit)))
    for eta, pt, expected in [(2.5,35.,1.),(-2.5,35.,1.),(np.nextafter(2.5,np.inf),35.,1.2),(3.,50.,1.2),(3.,70.,1.2),(3.,np.nextafter(100.,0.),1.2),(3.,100.,1.),(-3.,35.,1.2)]:
        assert cset['jet_sf'].evaluate(1,pt,eta) == expected
        assert cset['jet_sf_error'].evaluate(1,pt,eta) == (0. if expected == 1. else 0.1)


def test_fixed_covariance():
    p=dict(jet_sf=[[1., 1.2]], parameter_slices={'jet':[0,2]}, inapplicable_parameters=[0], optimizer_converged=True)
    info=information_covariance(np.diag([0.,4.]), [False,True])
    c=update_payload(p,np.log([1.,1.2]),info,info)
    assert p['errors_valid'] and p['jet_sf_error'][0][0] == 0.
    assert p['jet_sf_error_valid'][0][0]
    np.testing.assert_array_equal(c[0],0.)
    np.testing.assert_array_equal(c[:,0],0.)
    np.testing.assert_array_equal(info["covariance"][0],0.)
    np.testing.assert_array_equal(info["covariance"][:,0],0.)


def test_w_fit_uses_event_product():
    from w_sf import N_W_SF, W_MUMET_COLUMN
    keys = np.full((3,117), 65535, np.uint16)
    keys[:,1] = [2,0,1]
    keys[:,2:4] = 0
    keys[:,4] = 0
    keys[:,37] = [0,65535,1]
    keys[0,38] = 0
    keys[:,W_MUMET_COLUMN] = 0
    sums = np.zeros((3,7)); sums[:,0] = [2.,-3.,4.]
    mc = design(keys,sums,np.zeros(3,int),2,1,'wcr',['W'],1,use_w_sf=True)
    x = np.zeros(2+1+N_W_SF); x[:2] = np.log([1.2,1.4]); x[3:] = np.log(1.5)
    _, jac, details = model_and_jacobian(x,mc,return_details=True)
    np.testing.assert_allclose(details['raw'],[1.2**2,1.,1.4])
    np.testing.assert_allclose(details['weights'],sums[:,0]*details['raw']*1.5)
    d=np.zeros_like(x); d[0]=1e-5
    np.testing.assert_allclose((model_and_jacobian(x+d,mc)[0]-model_and_jacobian(x-d,mc)[0])/2e-5,jac[:,0],rtol=1e-8,atol=1e-8)


def test_preflight_uses_new_dimensions(monkeypatch):
    from scipy import sparse
    import campaign
    import fit_model
    import conditional_model
    config.configure_reference('2024')
    manifest = json.loads((CORE / 'model.json').read_text())
    manifest.update(eras=['2024'], configuration={'selection': config.selection_metadata()})
    monkeypatch.setattr(campaign, 'read', lambda path: manifest, raising=False)
    monkeypatch.setattr(campaign, 'RUN', _run, raising=False)
    def dimension(region):
        return config.N_ZCR_OBS if region == 'zee' else config.N_WCR_OBS + 4 + 10 + 104
    def context(run, manifest, era, region):
        n = dimension(region)
        return {'h': sparse.csr_matrix(np.ones((1, n))), 'sumw': np.array([2.])}
    monkeypatch.setattr(fit_model, 'load_context', context)
    monkeypatch.setattr(fit_model, 'load_data', lambda run, manifest, era, region: context(run, manifest, era, region))
    monkeypatch.setattr(conditional_model, 'model_and_jacobian',
                        lambda x, mc: (np.full(mc['h'].shape[1], 2.), np.zeros((mc['h'].shape[1], len(x)))) )
    campaign.preflight('2024')


def test_all_strict_boundaries_and_both_match_types():
    config.configure_reference('2024')
    model = json.loads((CORE / 'model.json').read_text())['model']
    manifest = dict(eras=['2024'], model=model, configuration={'selection': {'jet_selection': 'inclusive'}})
    shape = (2, config.N_PT, config.N_ETA)
    fixed = np.zeros(config.N_JET_CELLS, bool)
    fixed[inapplicable_candidates(manifest)] = True
    jet = np.where(fixed, 1., 1. + (np.arange(len(fixed)) + 1) / 10000.)
    fit = dict(eras=['2024'], optimizer_converged=True, fit_success=True, errors_valid=True,
        jet_sf=jet.reshape(shape).tolist(), jet_sf_error=np.where(fixed, 0., .1).reshape(shape).tolist(),
        jet_sf_error_valid=np.ones(shape, bool).tolist(), jet_sf_near_boundary=np.zeros(shape, bool).tolist(),
        jet_sf_fixed=fixed.reshape(shape).tolist(), jet_sf_applicable=np.ones(shape, bool).tolist(),
        dy_sf=np.ones((26, 5)).tolist(), dy_sf_error=np.full((26, 5), .1).tolist(),
        dy_sf_error_valid=np.ones((26, 5), bool).tolist(), dy_sf_near_boundary=np.zeros((26, 5), bool).tolist(),
        dy_sf_applied_regions=['zee', 'zcr', 'srsb'])
    cset = correctionlib.CorrectionSet.from_string(json.dumps(build_payload(manifest, fit)))
    pts = [v for edge in (50., 70., 100.) for v in (np.nextafter(edge, -np.inf), edge, np.nextafter(edge, np.inf))]
    etas = [3., -3.] + [v for edge in (-2.5, 2.5) for v in (np.nextafter(edge, -np.inf), edge, np.nextafter(edge, np.inf))]
    for matched in (0, 1):
        for pt in pts:
            for eta in etas:
                a = ak.Array(dict(Jet_corrPt=[[pt]], Jet_corrEta=[[eta]], Jet_corrPhi=[[0.]],
                    Jet_corrMass=[[0.]], Jet_isGenMatched=[[matched]],
                    zee_pt=[30.], zee_phi=[0.], zee_mass=[91.]))
                jets = selected_jets(a, False)
                cell = int(ak.to_list(jets[4])[0][0])
                applicable = abs(eta) > 2.5 and pt < 100.
                expected = jet[cell] if applicable else 1.
                factors = event_factors(jets, dict(sample='DY', group='DY', region='zee', dy_sf_applicable=True), fit, np.array([30.]))
                np.testing.assert_allclose(factors[0, 1], expected, rtol=1e-14)
                key = signatures(a, 'zee', True)[0, 37]
                assert key == (cell if applicable else 65535)
                np.testing.assert_allclose(cset['jet_sf'].evaluate(matched, pt, eta), expected, rtol=1e-14)
                assert cset['jet_sf_error'].evaluate(matched, pt, eta) == (.1 if applicable else 0.)
