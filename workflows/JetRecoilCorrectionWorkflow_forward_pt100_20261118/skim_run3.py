#!/usr/bin/env python3
"""Frozen Run3 correction + minimal skim production. No training or inference."""
import argparse
import ast
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zlib

CORE = Path(__file__).resolve().parent
BASE = Path('/eos/user/j/jiehan/HMuMuShare')
OLD = BASE / 'JetRecoilCorrection_Run3/20261101T143656Z_zee_wcr'
ERAS = ('2022', '2022EE', '2023', '2023BPix', '2024', '2025', '2026')
PYTHON = '/eos/user/j/jiehan/ml-gpu-env/bin/python3'
PROXY = '/afs/cern.ch/user/j/jiehan/x509up_u152815'
sys.path[:0] = [str(CORE), str(CORE/'scripts/Jet_RecoilCorrection'), str(CORE/'scripts')]


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.partial')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def current_path(p, era, region):
    p = Path(p)
    parts = p.parts
    i = next(i for i, part in enumerate(parts) if part.startswith(region.upper()+'_'))
    return BASE / (region.upper()+'_'+era) / Path(*parts[i+1:])


def file_identity(p):
    s = Path(p).stat()
    return dict(path=str(p), size=s.st_size, mtime_ns=s.st_mtime_ns)


def inspect(p, era, data):
    import numpy as np
    import uproot
    with uproot.open(p, object_cache=None, array_cache=None) as f:
        t = f['ntuple']
        result = file_identity(p)
        result.update(entries=t.num_entries, branches=t.keys(),
                      schema_sha256=hashlib.sha256(json.dumps(t.typenames(), sort_keys=True).encode()).hexdigest(),
                      production_fingerprint=str(f['production_fingerprint']),
                      output_schema=str(f['output_schema']))
        if not data:
            name = 'normalization_sumw' if era in ('2024','2025','2026') else 'genEventSumw'
            denom = float(f['conditions'][name].array(library='np').sum(dtype=np.float64))
            if not np.isfinite(denom) or denom == 0:
                raise ValueError(f'Invalid {name}: {p}')
            result.update(normalization_branch=name, normalization_sumw=denom)
        return result


def literal_assignment(path, name):
    for n in ast.parse(Path(path).read_text()).body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return ast.literal_eval(n.value)
    raise KeyError(name)


def prepare(run, control):
    if run.exists() or control.exists():
        raise FileExistsError('Production and control directories must be new')
    run.mkdir(parents=True)
    control.mkdir(parents=True)
    training = Path('/afs/cern.ch/user/j/jiehan/private/ml_hmm/hzgml/data/training_config_BDT_ggH_multiyear_template.py')
    xs_path = Path('/afs/cern.ch/user/j/jiehan/private/ml_hmm/hzgml/scripts/Jet_DNN/until.py')
    xs = literal_assignment(xs_path, 'XSECTIONS')
    aliases = literal_assignment('/afs/cern.ch/user/j/jiehan/private/recoil_workflow_run3/build_inputs.py', 'aliases')
    settings = read(CORE/'eras.json')
    oldcat = read(CORE/'samples.json')['eras']
    records = []
    tasks = []
    for era in ERAS:
        for region in ('srsb', 'zcr'):
            folder = BASE / (region.upper()+'_'+era)
            for sample, paths in [('data', sorted((folder/'data').glob('*.root')))]+[(p.stem,[p]) for p in sorted((folder/'mc').glob('*.root'))]:
                if not paths:
                    raise ValueError(f'Empty sample {folder}/{sample}')
                data = sample == 'data'
                signal = not data and 'Hto2Mu' in sample
                known = next((r for r in oldcat[era][region] if r['sample'] == sample), None)
                xsec = None if data else (known['cross_section_pb'] if known else xs[aliases.get(sample, sample)])
                r = dict(index=len(records), era=era, region=region, sample=sample, data=data,
                         signal=signal, cross_section_pb=xsec, lumi_pb=settings[era]['lumi_pb'],
                         dy=sample.startswith('DY') and not sample.startswith('DYto2Tau'),
                         output=f'{era}/{region.upper()}/{sample}.root', inputs=[])
                records.append(r)
                tasks.extend((r, p) for p in paths)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        fut = [(r, pool.submit(inspect, p, r['era'], r['data'])) for r,p in tasks]
        for k,(r,f) in enumerate(fut):
            info = f.result()
            if info['production_fingerprint'] != settings[r['era']]['production_fingerprint']:
                raise ValueError(f'Unexpected source production: {info["path"]}')
            r['inputs'].append(info)
            if k % 25 == 0:
                print('INSPECT', k, len(fut), flush=True)
    for r in records:
        branches = set(r['inputs'][0]['branches'])
        if any(set(i['branches']) != branches for i in r['inputs']):
            raise ValueError('Mixed sample schemas: '+r['sample'])
        r['shifts'] = sorted({k.split('__',1)[1] for k in branches if k.startswith('Jet_selectionMask__')}) if r['signal'] else []
        r['normalization_sumw'] = None if r['data'] else sum(i['normalization_sumw'] for i in r['inputs'])
        r['scale'] = 1. if r['data'] else r['lumi_pb']*r['cross_section_pb']/r['normalization_sumw']
    config = literal_assignment(training, 'CONFIG')['ggH']
    manifest = dict(eras=list(ERAS), records=records, training_variables=config['train_variables'],
                    source_fit=str(OLD), control_dir=str(control), tree='Events', max_jobs=4,
                    training_config_sha256=digest(training), cross_sections_sha256=digest(xs_path),
                    schema_version=1, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    write(run/'manifest.json', manifest)
    write(run/'completion.json', dict(completed=False, stage='prepared'))
    shutil.copytree(CORE, run/'code', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copyfile(training, run/'reference_training_config.py')
    hashes = {str(p.relative_to(run/'code')):digest(p) for p in (run/'code').rglob('*') if p.is_file()}
    write(run/'code_sha256.json', hashes)
    print('PREPARED',run,len(records),flush=True)


def check_record(r):
    for i in r['inputs']:
        identity = file_identity(i['path'])
        if any(identity[k] != i[k] for k in ('size','mtime_ns')):
            raise ValueError('Source changed: '+i['path'])


def sf_variance(indices, covariance):
    """Log-product variance; repeated indices are perfectly correlated."""
    import awkward as ak
    import numpy as np
    n = len(indices)
    width = int(ak.max(ak.num(indices), initial=0))
    if not width:
        return np.zeros(n)
    p = ak.to_numpy(ak.fill_none(ak.pad_none(indices, width, clip=True), -1))
    result = np.zeros(n)
    absolute = np.zeros(n)
    for j in range(width):
        for k in range(j, width):
            good = (p[:,j] >= 0) & (p[:,k] >= 0)
            term = covariance[p[good,j], p[good,k]] * (1 if j == k else 2)
            result[good] += term
            absolute[good] += abs(term)
    if np.any(result < -1e-11*np.maximum(absolute, 1e-30)):
        raise ValueError('Negative SF variance')
    return np.maximum(result, 0)


def category(a, values, region, mass):
    import numpy as np
    good = (np.asarray(a['nbjets_medium']) == 0) & (np.asarray(a['nbjets_loose']) <= 1)
    good &= np.asarray(a['trg_single_mu24'],bool) & (np.asarray(a['nmuons']+a['nelectrons']) < 3)
    good &= np.isfinite(mass) & np.isfinite(values['diMufsr_kit_BSC_pt'])
    good &= ((mass > 110)&(mass < 150)) if region == 'srsb' else ((mass > 70)&(mass < 110))
    vbf = (values['njets'] >= 2)&(values['jet1_pt'] >= 35)&(values['jet2_pt'] >= 25)&(values['delta_eta_jj'] >= 2.5)&(values['dijet_mass'] >= 400)
    return np.where(good & vbf, 3, np.where(good & (region == 'srsb'), 4, 0)).astype('u1')


def resolution(a):
    import numpy as np
    terms = []
    for i in (1,2):
        use = np.asarray(a[f'mu{i}_fromH_bsConstrainedChi2']) < 30
        pt = np.where(use, np.asarray(a[f'mu{i}_fromH_bsConstrainedPt']), np.asarray(a[f'mu{i}_fromH_pt']))
        err = np.where(use, np.asarray(a[f'mu{i}_fromH_bsConstrainedPtErr']), np.asarray(a[f'mu{i}_fromH_ptErr']))
        terms.append((err/pt)**2)
    return .5*np.sqrt(terms[0]+terms[1])


def corrections(run, era):
    import numpy as np
    p = run/'corrections'/era
    f = read(p/'fit_results.json')
    if not all(f[k] for k in ('optimizer_converged','fit_success','errors_valid')):
        raise ValueError('Invalid correction fit: '+era)
    if not f['history'][-1]['scaled_gradient_max'] < f.get('scaled_gradient_tolerance', 1e-5):
        raise ValueError('Fit stationarity failed: '+era)
    return f, np.load(p/'covariance.npy')


def build_chunk(a, r, fit, covariance, variables):
    import awkward as ak
    import numpy as np
    from scipy import sparse
    from skim_features import features, REQUIRED_JETS
    from until import selected_jets, dilepton_kinematics
    import config

    jets = selected_jets(a, r['data'])
    values, valid = features(a, jets, r['era'])
    mass = dilepton_kinematics(a)[2]
    cate = category(a, values, r['region'], mass)
    jet_sf = np.ones(len(a)) if r['data'] else ak.to_numpy(ak.prod(ak.unflatten(np.asarray(fit['jet_sf']).ravel()[ak.to_numpy(ak.flatten(jets[4]))], ak.num(jets[4])), axis=1))
    dy_sf = np.ones(len(a))
    if r['dy']:
        ptbin = np.searchsorted(config.DY_PT_EDGES, values['diMufsr_kit_BSC_pt'],side='right')-1
        if np.any((ptbin < 0)|(ptbin >= len(config.DY_PT_EDGES)-1)):
            raise ValueError('Invalid DY pT bin')
        idx = ptbin*config.DY_NJET_CATEGORIES+np.minimum(values['njets'],config.DY_NJET_CATEGORIES-1)
        dy_sf = np.asarray(fit['dy_sf']).ravel()[idx]
    if r['data']:
        raw = np.ones(len(a)); id_sf=iso_sf=pu=np.ones(len(a))
    else:
        raw = np.asarray(a['Generator_weight'],float)*r['scale']
        id_sf = np.asarray(a['id_wgt_mu_1'],float)*np.asarray(a['id_wgt_mu_2'],float)
        iso_sf = np.asarray(a['iso_wgt_mu_1'],float)*np.asarray(a['iso_wgt_mu_2'],float)
        pu = np.asarray(a['puweight'],float)
    event_weight = raw*id_sf*iso_sf*pu*jet_sf*dy_sf
    out = {k:np.where(valid[k], values[k], -100).astype('f4') for k in variables}
    out.update(diMufsr_kit_BSC_mass=mass.astype('f8'), **{'diMu-mass_resolution_BSC':resolution(a).astype('f4')})
    out.update({k:ak.to_numpy(a[k]) for k in ('run','lumi','event')})
    out.update(weight=np.full(len(a),r['scale'],'f8'), eventWeight=np.where(cate>0,event_weight,0).astype('f8'), cate_index=cate)
    keep = cate > 0
    if r['signal']:
        out.update(jet_sf_norm=jet_sf, id_sf_norm=id_sf, iso_sf_norm=iso_sf, pu_sf_norm=pu,
                   PDF_norm=np.ones(len(a)), QCD_norm=np.ones(len(a)))
        sigma = jet_sf*np.sqrt(sf_variance(jets[4], covariance[:560,:560]))
        for direction, sign in (('up',1),('down',-1)):
            out['jet_sf_'+direction] = jet_sf + sign*sigma
            out['eventWeight__JetSF'+direction] = np.where(cate>0, raw*id_sf*iso_sf*pu*(jet_sf+sign*sigma), 0.)
            idvar=np.asarray(a['id_wgt_mu_1__MuonEff'+direction],float)*np.asarray(a['id_wgt_mu_2__MuonEff'+direction],float)
            isovar=np.asarray(a['iso_wgt_mu_1__MuonEff'+direction],float)*np.asarray(a['iso_wgt_mu_2__MuonEff'+direction],float)
            out['id_sf_'+direction],out['iso_sf_'+direction] = idvar,isovar
            out['eventWeight__MuonEff'+direction] = np.where(cate>0,raw*idvar*isovar*pu*jet_sf,0.)
            for name, source in (('PDF','PDF_uncertainty_'),('QCD','qcd_unc_')):
                factor=np.asarray(a[source+direction],float)
                out[name+'_'+direction]=factor
                out['eventWeight__'+name+direction]=np.where(cate>0,event_weight*factor,0.)
        counts = ak.to_numpy(ak.num(jets[4])); flat=ak.to_numpy(ak.flatten(jets[4])).astype(int)
        matrix=sparse.csr_matrix((np.ones(len(flat),'u1'),(np.repeat(np.arange(len(a)),counts),flat)),shape=(len(a),560))
        matrix.sum_duplicates();matrix.sort_indices()
        out['sf_index']=ak.unflatten(matrix.indices.astype('i2'),np.diff(matrix.indptr))
        out['sf_count']=ak.unflatten(matrix.data.astype('u1'),np.diff(matrix.indptr))
        for shift in r['shifts']:
            fields={k:a[k] for k in a.fields if '__' not in k}
            suffix='__'+shift
            fields.update({k[:-len(suffix)]:a[k] for k in a.fields if k.endswith(suffix)})
            shifted=ak.Array(fields)
            js=selected_jets(shifted,False)
            vs,oks=features(shifted,js,r['era'])
            cs=category(shifted,vs,r['region'],mass)
            keep |= cs>0
            sf=ak.to_numpy(ak.prod(ak.unflatten(np.asarray(fit['jet_sf']).ravel()[ak.to_numpy(ak.flatten(js[4]))],ak.num(js[4])),axis=1))
            for k in variables:
                if k in REQUIRED_JETS or k == 'njets':
                    out[k+suffix]=np.where(oks[k],vs[k],-100).astype('f4')
            out['cate_index'+suffix]=cs
            out['eventWeight'+suffix]=np.where(cs>0,raw*id_sf*iso_sf*pu*sf,0.)
    return {k:v[keep] for k,v in out.items()},dict(read=len(a), kept=int(keep.sum()), nominal=int((cate>0).sum()),
                   ggh=int((cate==4).sum()),vbf=int((cate==3).sum()),sumw=float(np.sum(out['eventWeight'][keep])))


def adler(p):
    result=1
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):
            result=zlib.adler32(chunk,result)
    return f'{result&0xffffffff:08x}'


def publish(local, destination):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    checksum=adler(local)
    temp=destination.with_name(destination.name+f'.upload.{os.getpid()}')
    # A completed, locally verified ROOT is the only large object uploaded to EOS.
    shutil.copyfile(local,temp)
    if adler(temp)!=checksum:
        temp.unlink();raise ValueError('Upload checksum mismatch')
    temp.replace(destination)
    return checksum


def schema_for(chunk):
    import awkward as ak
    return {k:('var * int16' if k=='sf_index' else 'var * uint8') if isinstance(v,ak.Array) else v.dtype for k,v in chunk.items()}


def validate_root(path, branches, expected=None):
    import numpy as np
    import uproot
    totals=dict(entries=0,sumw=0.,ggh=0,vbf=0)
    with uproot.open(path,object_cache=None,array_cache=None) as f:
        t=f['Events']
        if set(t.keys()) != set(branches):
            raise ValueError('Unexpected output branches: '+str(set(t.keys())^set(branches)))
        totals['entries']=t.num_entries
        for a in t.iterate(['eventWeight','cate_index'],step_size=500000,library='np'):
            if not np.isfinite(a['eventWeight']).all():raise ValueError('Nonfinite output weight')
            totals['sumw']+=float(a['eventWeight'].sum());totals['ggh']+=int((a['cate_index']==4).sum());totals['vbf']+=int((a['cate_index']==3).sum())
        if expected:
            if totals['entries']!=expected['kept'] or not np.isclose(totals['sumw'],expected['sumw'],rtol=1e-11,atol=1e-8):
                raise ValueError('Output event/yield mismatch')
    return totals


def worker(run,index,limit=0):
    import numpy as np
    import awkward as ak
    import uproot
    import config
    from Jet_DNN.jet_io import jet_tree
    manifest=read(run/'manifest.json');r=manifest['records'][index];check_record(r)
    config.configure(r['era']);fit,cov=corrections(run,r['era'])
    fit_sha=digest(run/'corrections'/r['era']/'fit_results.json')
    dest=run/('pilot' if limit else 'roots')/r['output']
    receipt=dest.with_suffix('.json')
    if receipt.exists() and dest.exists():
        saved=read(receipt)
        if saved['fit_sha256']==fit_sha and saved['record']==r and saved['limit']==limit and adler(dest)==saved['adler32']:
            print('VERIFIED_REUSE',dest,flush=True);return
        raise ValueError('Existing output has a different production identity')
    branches=list(config.ZCR_BRANCHES)
    branches += [f'mu{i}_fromH_{s}' for i in (1,2) for s in ('bsConstrainedChi2','bsConstrainedPt','bsConstrainedPtErr','pt','ptErr')]
    if not r['data']:branches+=['Jet_isGenMatched',*config.JET_WEIGHT_BRANCHES]
    if r['signal']:
        branches += [f'{kind}_wgt_mu_{i}__MuonEff{d}' for kind in ('id','iso') for i in (1,2) for d in ('up','down')]
        branches += ['PDF_uncertainty_up','PDF_uncertainty_down','qcd_unc_up','qcd_unc_down']
        for shift in r['shifts']:
            branches += [k+'__'+shift for k in ('Jet_corrPt','Jet_corrEta','Jet_corrPhi','Jet_corrMass','Jet_originalIndex','Jet_btagLoose','Jet_btagMedium','Jet_isGenMatched','nbjets_loose','nbjets_medium')]
    branches=list(dict.fromkeys(branches))
    totals=dict(read=0,kept=0,nominal=0,ggh=0,vbf=0,sumw=0.)
    began=time.monotonic();output_branches=None
    scratch=Path(os.environ.get('_CONDOR_SCRATCH_DIR',tempfile.gettempdir()))
    with tempfile.TemporaryDirectory(prefix='hmm_skim_',dir=scratch) as work:
        local=Path(work)/'final.root'
        with uproot.recreate(local,compression=uproot.ZLIB(4)) as output:
            for item in r['inputs']:
                with uproot.open(item['path'],object_cache=None,array_cache=None) as source:
                    tree=jet_tree(source['ntuple'])
                    stop=min(item['entries'],max(0,limit-totals['read'])) if limit else None
                    if limit and stop<=0:break
                    for a in tree.iterate(branches,step_size=20000 if r['signal'] else 200000,entry_stop=stop,library='ak'):
                        chunk,counts=build_chunk(a,r,fit,cov,manifest['training_variables'])
                        for name,value in chunk.items():
                            if isinstance(value,np.ndarray) and value.dtype.kind=='f' and not np.isfinite(value).all():
                                raise ValueError('Nonfinite output: '+name)
                        if output_branches is None:
                            output.mktree('Events',schema_for(chunk))
                            output_branches=list(output['Events'].keys()) if hasattr(output['Events'],'keys') else list(chunk)+(['nsf_index','nsf_count'] if r['signal'] else [])
                        if counts['kept']:output['Events'].extend(chunk)
                        for k,v in counts.items():totals[k]+=v
                        if totals['read']%1000000<200000:print('SKIM',index,r['era'],r['sample'],totals,flush=True)
            output['skim_metadata']=json.dumps(dict(record=r,fit_sha256=fit_sha,limit=limit,schema_version=1))
        if output_branches is None:raise ValueError('No input entries')
        verified=validate_root(local,output_branches,totals)
        checksum=publish(local,dest)
        size=dest.stat().st_size
        write(receipt,dict(record=r,fit_sha256=fit_sha,limit=limit,adler32=checksum,bytes=size,totals=totals,
                           verified=verified,branches=output_branches,elapsed_s=time.monotonic()-began))
    print('PUBLISHED',dest,size,totals,flush=True)


def fit_job(run,era):
    """Read old MC signatures; regenerate changed data signatures only in scratch."""
    import numpy as np
    os.environ['RECOIL_RUN_DIR']=str(OLD)
    import config
    import campaign
    import runtime
    import fit_model
    from types import SimpleNamespace
    from sf_export import export_sf
    os.environ['RECOIL_EXECUTION_TOOL_ID']=digest(Path(fit_model.__file__))
    old_manifest=read(OLD/era/'E/run_manifest.json')
    with tempfile.TemporaryDirectory(prefix='hmm_fit_',dir=os.environ.get('_CONDOR_SCRATCH_DIR',tempfile.gettempdir())) as work:
        work=Path(work);local=work/era/'E';local.mkdir(parents=True)
        m=json.loads(json.dumps(old_manifest));m['shared_cache_dir']=str(work/'shared')
        m['initial_parameters']=str(OLD/era/'E/fit/parameters.npy');m['refine_stationarity']=True
        m['stable_summation']=True
        records=[]
        for original in m['samples']:
            region=original['region'];sample=original['sample']
            declared=Path(old_manifest.get('shared_cache_dir',str(OLD/'shared')))/'cache'/era/region/(sample+'__all.npz')
            oldcache=declared if declared.exists() else OLD/'shared/cache'/era/region/(sample+'__all.npz')
            cache=work/'shared/cache'/era/region/(sample+'__all.npz');cache.parent.mkdir(parents=True,exist_ok=True)
            changed=False
            for item in original['inputs']:
                now=current_path(item['path'],era,region)
                if not now.exists():raise FileNotFoundError(now)
                changed |= now.stat().st_size!=item['size'] or now.stat().st_mtime_ns!=item['mtime_ns']
            cached=read(oldcache.with_suffix('.json'))
            def normalize_inputs(inputs):
                result=json.loads(json.dumps(inputs))
                for i in result:i['path']=str(current_path(i['path'],era,region))
                return result
            changed |= normalize_inputs(cached['inputs'])!=normalize_inputs(original['inputs'])
            if changed and sample!='data':raise ValueError('Changed fitted MC requires a new scan: '+sample)
            if changed:
                original['inputs']=[file_identity(p) for p in sorted((BASE/(region.upper()+'_'+era)/'data').glob('*.root'))]
                records.append(original)
            else:
                cache.symlink_to(oldcache)
                # Only the verified directory rename may differ. Keep this
                # adapted receipt local; never rewrite historical cache metadata.
                cached['inputs']=original['inputs']
                write(cache.with_suffix('.json'),cached)
        write(local/'run_manifest.json',m)
        for rec in records:
            # Data scales are unity; the scan kernel uses no source MC catalog lookup.
            rec['index']=records.index(rec)
        write(work/'inputs.json',dict(records=records))
        write(work/era/'E/run_manifest.json',m)
        campaign.RUN=work
        # campaign normally installs these helpers in its CLI entry point.
        for name in ('read','write','save','sha'):
            setattr(campaign,name,getattr(runtime,name))
        for index,record in enumerate(records):campaign.scan(index)
        fit_model.fit(SimpleNamespace(run_dir=local,era=era),m)
        f=read(local/'fit/fit_results.json')
        if not all(f[k] for k in ('optimizer_converged','errors_valid','fit_success')):raise RuntimeError('Fit did not pass: '+era)
        export_sf(local)
        target=run/'corrections'/era;target.mkdir(parents=True,exist_ok=True)
        for p in (local/'fit').iterdir():
            if p.is_file():shutil.copyfile(p,target/p.name)
        metadata_path=target/'corrections_metadata.json'
        if metadata_path.exists():
            metadata=read(metadata_path)
            for name,info in metadata.get('covariance_files',{}).items():info['path']=str(target/name)
            write(metadata_path,metadata)
        write(target/'input_update.json',dict(source_fit=str(OLD/era/'E/fit'),regenerated_data=records,
                    manifest=m,fit_model_sha256=digest(Path(fit_model.__file__)),fit_results_sha256=digest(target/'fit_results.json')))
    print('FIT_PUBLISHED',era,flush=True)


def stage_existing(run):
    for era in ERAS:
        if era in ('2025','2026'):continue
        source=OLD/era/'E/fit';fit=read(source/'fit_results.json')
        if not all(fit[k] for k in ('optimizer_converged','errors_valid','fit_success')):raise ValueError('Invalid source fit '+era)
        # Fit input identities must match after the declared directory rename.
        m=read(OLD/era/'E/run_manifest.json')
        for r in m['samples']:
            for i in r['inputs']:
                now=current_path(i['path'],era,r['region'])
                if now.stat().st_size!=i['size'] or now.stat().st_mtime_ns!=i['mtime_ns']:
                    raise ValueError('Source fit input changed: '+str(now))
        target=run/'corrections'/era;target.mkdir(parents=True,exist_ok=True)
        for p in source.iterdir():
            if p.is_file() and (p.suffix=='.npy' or p.name in ('fit_results.json','corrections.json','corrections.json.gz','corrections_metadata.json')):
                shutil.copyfile(p,target/p.name)
        write(target/'input_update.json',dict(reused=True,source_fit=str(source),directory_mapping='REGION_ERA_nohehf_* -> REGION_ERA',verified_size_and_mtime=True))
        corrections(run,era)


def pilot_indices(manifest):
    result=[]
    for era in ERAS:
        for region,prefixes in [('srsb',('data','DY','GluGluH','VBFH')),('zcr',('data','DY'))]:
            for prefix in prefixes:
                result.append(next(r['index'] for r in manifest['records'] if r['era']==era and r['region']==region and r['sample'].startswith(prefix)))
    for prefix in ('TTH','WminusH','WplusH','ZH'):
        result.append(next(r['index'] for r in manifest['records'] if r['era']=='2024' and r['region']=='srsb' and r['sample'].startswith(prefix)))
    return result


def pilot_check(run):
    import numpy as np
    import awkward as ak
    import uproot
    from scipy import sparse
    from skim_covariance import accumulate,bin_covariance
    manifest=read(run/'manifest.json');results=[]
    for index in pilot_indices(manifest):
        r=manifest['records'][index];p=run/'pilot'/r['output'];receipt=read(p.with_suffix('.json'))
        result=validate_root(p,receipt['branches'],receipt['totals'])
        if receipt['limit']!=20000:raise ValueError('Pilot entry limit differs')
        with uproot.open(p) as f:
            arrays=f['Events'].arrays(library='ak')
            for name in arrays.fields:
                if name.startswith(('nsf_','sf_')):continue
                if np.asarray(arrays[name]).dtype.kind=='f' and not np.isfinite(np.asarray(arrays[name])).all():raise ValueError('Nonfinite pilot field '+name)
            if not r['signal']:
                if any('__' in k or k.startswith(('jet_sf_','id_sf_','iso_sf_','sf_index')) for k in f['Events'].keys()):raise ValueError('Background has systematic columns')
            else:
                from skim_features import REQUIRED_JETS
                from Jet_DNN.jet_io import jet_tree
                for shift in r['shifts']:
                    required={k+'__'+shift for k in (*REQUIRED_JETS,'njets')}
                    if not required.issubset(arrays.fields):
                        raise ValueError('Missing shifted training features: '+str(required-set(arrays.fields)))
                # Independently count accepted jets from the logical source view,
                # including entries retained solely by a systematic migration.
                if len(r['inputs']) != 1:
                    raise ValueError('Signal pilot jet check expects one frozen input')
                names=['run','lumi','event']+[k+s for s in ['']+['__'+x for x in r['shifts']]
                                             for k in ('Jet_corrPt','Jet_corrEta')]
                with uproot.open(r['inputs'][0]['path'],object_cache=None,array_cache=None) as src:
                    original=jet_tree(src['ntuple']).arrays(names,entry_stop=20000,library='ak')
                key=lambda a:np.rec.fromarrays([np.asarray(a[k]) for k in ('run','lumi','event')],names='run,lumi,event')
                source_ids,output_ids=key(original),key(arrays)
                selected=np.isin(source_ids,output_ids)
                np.testing.assert_array_equal(source_ids[selected],output_ids)
                for suffix in ['']+['__'+x for x in r['shifts']]:
                    pt,eta=original['Jet_corrPt'+suffix],original['Jet_corrEta'+suffix]
                    counted=np.minimum(ak.to_numpy(ak.sum(np.isfinite(pt)&np.isfinite(eta)&(pt>25)&(abs(eta)<4.7),axis=1)),32)
                    np.testing.assert_array_equal(np.asarray(arrays['njets'+suffix]),counted[selected])
                fit,cov=corrections(run,r['era'])
                idx=arrays['sf_index'];counts=arrays['sf_count'];length=ak.to_numpy(ak.num(idx))
                design=sparse.csr_matrix((ak.to_numpy(ak.flatten(counts)),(np.repeat(np.arange(len(arrays)),length),ak.to_numpy(ak.flatten(idx)))),shape=(len(arrays),794)).toarray()
                theta=np.load(run/'corrections'/r['era']/'parameters.npy')
                central=np.exp(design@theta)
                np.testing.assert_allclose(central,np.asarray(arrays['jet_sf_norm']),rtol=2e-13,atol=1e-14)
                variance=np.einsum('ij,ij->i',design@cov,design)
                sigma=central*np.sqrt(np.maximum(variance,0))
                np.testing.assert_allclose(np.asarray(arrays['jet_sf_up']),central+sigma,rtol=2e-12,atol=1e-14)
                np.testing.assert_allclose(np.asarray(arrays['jet_sf_down']),central-sigma,rtol=2e-12,atol=1e-14)
                bins=np.minimum(np.asarray(arrays['njets']).astype(int),5)
                moments=accumulate(arrays,bins,6)
                keep=np.asarray(arrays['cate_index'])>0;weights=np.asarray(arrays['eventWeight'])[keep]
                expected=(weights[:,None]*design[keep]).sum(axis=0)
                np.testing.assert_allclose(moments['gradient'].sum(axis=0),expected,rtol=2e-12,atol=1e-10)
                binned=bin_covariance(moments['gradient'],cov)
                np.testing.assert_allclose(binned.sum(),expected@cov@expected,rtol=1e-11,atol=1e-10)
                active=np.flatnonzero(expected)
                if len(active):
                    j=active[len(active)//2];eps=1e-5
                    numeric=(weights@np.exp(eps*design[keep,j])-weights@np.exp(-eps*design[keep,j]))/(2*eps)
                    np.testing.assert_allclose(numeric,expected[j],rtol=2e-7,atol=1e-9)
        results.append(dict(index=index,**result))
    write(run/'pilot_validation.json',dict(passed=True,results=results))
    print('PILOT_CHECK_PASSED',len(results),flush=True)


def training_config(run):
    config=literal_assignment(run/'reference_training_config.py','CONFIG')['ggH']
    manifest=read(run/'manifest.json');yearly=[]
    for era in ERAS:
        records=[r for r in manifest['records'] if r['era']==era and r['region']=='srsb']
        mapping={}
        for name,prefix in [('GluGluHToMuMu_M125','GluGluH'),('VBFHToMuMu_M125','VBFH'),('DY_105To160_ZpT-reweighted','DY'),('TTTo2L2Nu','TTto'),('ST_tW_top','TWminus'),('ST_tW_antitop','TbarW'),('WWTo2L2Nu','WWto'),('WZTo2L2Q','WZto2L2Q'),('WZTo3LNu','WZto3LNu'),('ZZTo2L2Nu','ZZto2L2Nu'),('ZZTo2L2Q','ZZto2L2Q'),('ZZTo4L','ZZto4L'),('WWW','WWW'),('WWZ','WWZ'),('WZZ','WZZ'),('ZZZ','ZZZ'),('data','data')]:
            matches=[r['sample'] for r in records if r['sample'].startswith(prefix)]
            if len(matches)!=1:raise ValueError(f'Ambiguous training map {era}/{name}: {matches}')
            mapping[name]=matches[0]
        ewk=[r['sample'] for r in records if r['sample'].startswith('EWK') and '105' in r['sample'] and 'herwig7' in r['sample']]
        if len(ewk)>1:raise ValueError('Ambiguous EWK training selection')
        if ewk:mapping['EWK_LLJJ_M105To160']=ewk[0]
        yearly.append(dict(year=era,folder=str(run/'roots'/era/'SRSB'),tree_name='Events',sample_map=mapping,
                           train_mc_background=[n for n in config['train_mc_background'] if n in mapping]))
    config['yearly_inputs']=yearly;config['preselections']=['diMufsr_kit_BSC_mass >= 110','diMufsr_kit_BSC_mass <= 150','cate_index == 4']
    config['keep_branches']=list(dict.fromkeys(config['keep_branches']+['cate_index']))
    (run/'training_config_skim.py').write_text('# Generated input config only; training has not been run.\nCONFIG = '+repr({'ggH':config})+'\n')


def finalize_correction_metadata(run,era):
    """Make exported correction provenance resolve inside the delivered bundle."""
    target=run/'corrections'/era;update=read(target/'input_update.json')
    archive=target/'run_manifest.json'
    if 'manifest' in update:
        write(archive,update['manifest'])
    else:
        source=Path(update['source_fit']).parent/'run_manifest.json'
        shutil.copyfile(source,archive)
    p=target/'corrections_metadata.json'
    metadata=read(p)
    metadata.setdefault('original_sources',json.loads(json.dumps(metadata['sources'])))
    for name,path in [('fit',target/'fit_results.json'),('manifest',archive)]:
        metadata['sources'][name]=dict(path=str(path),sha256=digest(path))
    for name,info in metadata['covariance_files'].items():
        path=target/name
        if digest(path)!=info['sha256']:raise ValueError('Correction covariance checksum mismatch')
        info['path']=str(path)
    write(p,metadata)


def finish(run):
    manifest=read(run/'manifest.json');summary=[]
    for r in manifest['records']:
        p=run/'roots'/r['output'];s=read(p.with_suffix('.json'));check_record(r)
        if s['record']!=r or s['limit']!=0 or s['fit_sha256']!=digest(run/'corrections'/r['era']/'fit_results.json') or adler(p)!=s['adler32']:
            raise ValueError('Invalid final receipt: '+str(p))
        if r['signal']:
            from skim_features import REQUIRED_JETS
            required={k+'__'+shift for shift in r['shifts'] for k in (*REQUIRED_JETS,'njets')}
            if not required.issubset(s['branches']):raise ValueError('Incomplete shifted training inputs: '+str(p))
        verified=validate_root(p,s['branches'],s['totals'])
        summary.append(dict(era=r['era'],region=r['region'],sample=r['sample'],bytes=p.stat().st_size,**verified))
    for era in ERAS:
        corrections(run,era)
        finalize_correction_metadata(run,era)
    if not read(run/'pilot_validation.json')['passed']:raise ValueError('Pilot validation has not passed')
    training_config(run)
    write(run/'validation_report.json',dict(passed=True,files=summary,total_bytes=sum(s['bytes'] for s in summary)))
    text=['# Run3 corrected minimal skim','', 'All ROOT files and all seven correction fits passed validation.','',
          '| Region | Files | ROOT size (GiB) |','|---|---:|---:|']
    for region in ('srsb','zcr'):
        selected=[s for s in summary if s['region']==region]
        text.append(f'| {region.upper()} | {len(selected)} | {sum(s["bytes"] for s in selected)/1024**3:.3f} |')
    text+=['','- ROOT TTree: `Events`; ggH selection: `cate_index == 4`; VBF: `cate_index == 3`.',
           '- Higgs signal includes available systematic norm/up/down and correlated Jet SF information.',
           '- Background/data contain central values only.',
           '- Read `SKIM_README.md` for migration flags and exact covariance propagation.',
           '- Input-only training configuration: `training_config_skim.py`.',
           '- No training or score inference has been run.','', '| Era | Fit success | Scaled gradient |','|---|---|---:|']
    for era in ERAS:
        f=read(run/'corrections'/era/'fit_results.json')
        text.append(f'| {era} | {f["fit_success"]} | {f["history"][-1]["scaled_gradient_max"]:.6g} |')
    (run/'REPORT.md').write_text('\n'.join(text)+'\n')
    # Trial ROOTs were bounded validation artifacts, not permanent output/cache.
    # The full final ROOTs contain these same entries; keep their JSON receipts.
    for folder in ('pilot','development_smoke'):
        for p in (run/folder).glob('*/*/*.root'):
            if p.is_file():p.unlink()
    write(run/'completion.json',dict(completed=True,stage='validated',files=len(summary),total_bytes=sum(s['bytes'] for s in summary)))
    print('PRODUCTION_COMPLETE',len(summary),sum(s['bytes'] for s in summary),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','stage-existing','fit','worker','pilot-check','await-fits','finish'])
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--control-dir',type=Path);p.add_argument('--era');p.add_argument('--index',type=int);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args();run=a.run_dir
    if a.stage=='prepare':prepare(run,a.control_dir)
    elif a.stage=='stage-existing':stage_existing(run)
    elif a.stage=='fit':fit_job(run,a.era)
    elif a.stage=='worker':worker(run,a.index,a.limit)
    elif a.stage=='pilot-check':pilot_check(run)
    elif a.stage=='await-fits':
        began=time.monotonic()
        while not all((run/'corrections'/era/'input_update.json').exists() for era in ERAS):
            if time.monotonic()-began>20*3600:raise TimeoutError('Correction fits have not completed')
            print('AWAIT_CORRECTIONS',flush=True);time.sleep(30)
        for era in ERAS:corrections(run,era)
    elif a.stage=='finish':finish(run)


if __name__=='__main__':main()
