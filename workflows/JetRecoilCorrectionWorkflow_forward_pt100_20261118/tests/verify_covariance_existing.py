from pathlib import Path
import sys, json, tempfile, shutil, hashlib, ast, os
import numpy as np
base=Path(__file__).resolve().parents[1]
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-dir', type=Path, required=True)
parser.add_argument('--report', type=Path, required=True)
args=parser.parse_args()
run=args.run_dir
src=base/'scripts/Jet_RecoilCorrection'
sys.path.insert(0,str(src))
from sf_covariance import recompute
from sf_export import export_sf, verify_export
# The actual optimizer and all scan/cache functions are unchanged.
old=(run/'code_snapshot/scripts/Jet_RecoilCorrection/fit_model.py').read_text()
new=(src/'fit_model.py').read_text()
assert old[old.index('def fit(args'):old.index('    data_hessian = parts[1]')]==new[new.index('def fit(args'):new.index('    data_hessian = parts[1]')]
old_defs={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(old).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name!='fit'}
new_defs={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(new).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name!='fit'}
assert old_defs==new_defs
results={}
for era in ('2022','2022EE','2023','2023BPix','2026'):
    original=run/era/'E/fit'
    before=json.loads((original/'fit_results.json').read_text())
    with tempfile.TemporaryDirectory(prefix='recoil_covariance_verify_',dir='/tmp/jiehan') as d:
        d=Path(d); f=d/'fit';f.mkdir()
        shutil.copy2(original.parent/'run_manifest.json',d/'run_manifest.json')
        for name in ('fit_results.json','parameters.npy','hessian.npy','data_hessian.npy'):
            shutil.copy2(original/name,f/name)
        after=recompute(f)
        assert after['errors_valid'] and after['regularized_rank']==794
        for name in ('jet_sf','dy_sf','w_sf','predictions','objective','chi2','sf_prior_chi2','history','optimizer_converged'):
            assert before[name]==after[name],(era,name)
        for name in ('jet','dy','w'):
            olderr=np.asarray(before[name+'_sf_error'],float)
            newerr=np.asarray(after[name+'_sf_error'],float)
            ok=np.isfinite(olderr)
            if before["errors_valid"]:
                np.testing.assert_allclose(newerr[ok],olderr[ok],rtol=1e-5,atol=1e-10)
            assert np.isfinite(newerr).all()
        cov=np.load(f/'covariance_sf.npy')
        np.testing.assert_allclose(cov,cov.T,rtol=1e-11,atol=1e-15)
        assert np.linalg.eigvalsh(cov).min()>-np.max(np.diag(cov))*1e-10
        export_sf(d);verify_export(d)
        errors=np.concatenate([np.asarray(after[n+'_sf_error']).ravel() for n in ('jet','dy','w')])
        np.testing.assert_allclose(np.diag(cov),errors**2,rtol=1e-14)
        results[era]=dict(rank=after['regularized_rank'],errors_valid=True,boundary=len(after['boundary_parameter_indices']),old_invalid=sum(not x for n in ('jet','dy','w') for x in np.asarray(before[n+'_sf_error_valid']).ravel()),recovered_errors={str(i):float(errors[i]) for i in (388,) if era=='2022'})
    print(era,results[era],flush=True)
args.report.write_text(json.dumps(results,indent=2)+'\n')
