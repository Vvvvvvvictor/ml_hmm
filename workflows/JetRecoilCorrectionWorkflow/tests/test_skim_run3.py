"""Independent covariance/selection regression checks for minimal production."""
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import awkward as ak
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skim_run3 import sf_variance, category, build_chunk


class SkimTest(unittest.TestCase):
    def test_shifted_njets_is_saved_with_category_migration(self):
        fields=dict(run=[1,1],lumi=[2,2],event=[3,4],Generator_weight=[-1.,2.],
                    nbjets_medium=[0,0],nbjets_loose=[0,0],trg_single_mu24=[1,1],
                    nmuons=[2,2],nelectrons=[0,0],puweight=[1.,1.])
        for kind in ('id','iso'):
            for mu in (1,2):
                for suffix in ('','__MuonEffup','__MuonEffdown'):
                    fields[f'{kind}_wgt_mu_{mu}'+suffix]=[1.,1.]
        for stem in ('PDF_uncertainty_','qcd_unc_'):
            for direction in ('up','down'):fields[stem+direction]=[1.,1.]
        def values(njets):
            v=dict(diMufsr_kit_BSC_pt=np.ones(2),njets=np.array(njets),
                   jet1_pt=np.full(2,50.),jet2_pt=np.full(2,30.),
                   delta_eta_jj=np.full(2,3.),dijet_mass=np.full(2,500.))
            return v,{k:np.ones(2,dtype=bool) for k in v}
        cells=[ak.Array([[0],[0,1]]),ak.Array([[0,1],[0]])]
        jets=[(ak.ones_like(c),)*4+(c,) for c in cells]
        r=dict(data=False,signal=True,dy=False,scale=1.,region='srsb',era='2024',shifts=['JERUp'])
        with patch('until.selected_jets',side_effect=jets), \
             patch('until.dilepton_kinematics',return_value=(None,None,np.full(2,125.))), \
             patch('skim_features.features',side_effect=[values([1,2]),values([2,1])]), \
             patch('skim_run3.resolution',return_value=np.ones(2)):
            out,_=build_chunk(ak.Array(fields),r,dict(jet_sf=np.ones(560)),np.zeros((560,560)),['njets'])
        np.testing.assert_array_equal(out['njets'],[1,2])
        np.testing.assert_array_equal(out['njets__JERUp'],[2,1])
        np.testing.assert_array_equal(out['cate_index'],[4,3])
        np.testing.assert_array_equal(out['cate_index__JERUp'],[3,4])
        np.testing.assert_array_equal(out['eventWeight__JERUp'],[-1.,2.])

    def test_shared_parameters_and_cross_event_covariance(self):
        c = np.array([[.04, -.012], [-.012, .09]])
        indices = ak.Array([[], [0], [0, 0], [0, 1]])
        p = np.array([[0.,0.], [1.,0.], [2.,0.], [1.,1.]])
        np.testing.assert_allclose(sf_variance(indices,c), np.einsum('ij,jk,ik->i',p,c,p))
        weights = np.array([1., -2., 3., 4.])
        g = weights @ p
        full = g @ c @ g
        event_quadrature = (weights**2) @ np.einsum('ij,jk,ik->i',p,c,p)
        self.assertGreater(abs(full-event_quadrature),.01)
        # Finite differences of the same signed total yield in log-SF coordinates.
        eps=1e-5
        numerical=np.array([(weights @ np.exp(p @ (np.eye(2)[i]*eps))-
                             weights @ np.exp(p @ (-np.eye(2)[i]*eps)))/(2*eps) for i in range(2)])
        np.testing.assert_allclose(numerical,g,rtol=1e-9)

    def test_category_edges_and_migration(self):
        a=ak.Array(dict(nbjets_medium=[0,0,1,0],nbjets_loose=[0,1,1,0],
                        trg_single_mu24=[1,1,1,1],nmuons=[2]*4,nelectrons=[0]*4))
        v=dict(diMufsr_kit_BSC_pt=np.ones(4),njets=np.array([2,2,2,0]),
               jet1_pt=np.array([35.,34.9,50.,0.]),jet2_pt=np.array([25.,25.,30.,0.]),
               delta_eta_jj=np.array([2.5,2.5,3.,0.]),dijet_mass=np.array([400.,400.,500.,0.]))
        np.testing.assert_array_equal(category(a,v,'srsb',np.full(4,125.)),[3,4,0,4])
        np.testing.assert_array_equal(category(a,v,'zcr',np.full(4,90.)),[3,0,0,0])
        np.testing.assert_array_equal(category(a,v,'srsb',np.full(4,110.)),[0]*4)


if __name__=='__main__':unittest.main()
