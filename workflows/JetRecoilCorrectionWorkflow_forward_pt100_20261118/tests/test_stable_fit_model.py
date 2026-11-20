"""Stable accumulation preserves the physical model and analytic derivatives."""
import os
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy import sparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/Jet_RecoilCorrection'))
from conditional_model import design,model_and_jacobian


class StableFitTest(unittest.TestCase):
    def test_agreement_and_finite_difference(self):
        old=os.environ.pop('RECOIL_STABLE_SUMMATION',None)
        try:
            keys=np.full((4,105),65535,np.uint16)
            keys[:,1]=[1,1,2,0];keys[:,2]=[0,1,0,1];keys[:,3]=[1,0,1,0]
            keys[:,4]=[0,1,0,1];keys[:,37]=[0,1,1,65535];keys[2,38]=1
            sums=np.zeros((4,7));sums[:,0]=[3.,-2.,5.,7.]
            mc=design(keys,sums,np.zeros(4,int),560,2,'zee',['DY'],130)
            x=np.zeros(690);x[:2]=[.1,-.2]
            y0,j0=model_and_jacobian(x,mc)
            os.environ['RECOIL_STABLE_SUMMATION']='1'
            y,j=model_and_jacobian(x,mc)
            np.testing.assert_allclose(y,y0,rtol=1e-14)
            np.testing.assert_allclose(j,j0,rtol=1e-14)
            for i in [0,1,560,561]:
                d=np.zeros(690);d[i]=1e-5
                numerical=(model_and_jacobian(x+d,mc)[0]-model_and_jacobian(x-d,mc)[0])/(2e-5)
                np.testing.assert_allclose(numerical,j[:,i],rtol=1e-8,atol=1e-9)
        finally:
            if old is None:os.environ.pop('RECOIL_STABLE_SUMMATION',None)
            else:os.environ['RECOIL_STABLE_SUMMATION']=old

    def test_signed_cancellation(self):
        h=sparse.csr_matrix(np.ones((3,1)))
        weights=np.array([1e16,1.,-1e16])
        self.assertEqual(float((h.T@weights)[0]),0.)
        self.assertEqual(float((h.T.astype(np.longdouble)@weights.astype(np.longdouble))[0]),1.)


if __name__=='__main__':unittest.main()
