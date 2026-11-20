"""Refresh the forward workflow using the published Run 3 merge reports."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

CORE = Path(__file__).resolve().parent
SHARE = Path('/eos/user/j/jiehan/HMuMuShare')
PROD = Path('/afs/cern.ch/user/j/jiehan/private/CROWN_run3/logs/nohehf_mc_20261115_v1')
AUDIT = SHARE.parent / 'CROWN_data_audits/20261112_full_data'
OLD = SHARE / 'JetRecoilCorrection_Run3/20261111_641e_forward_pt80'
SUPPLEMENT = PROD / 'ewk_wcr_supplement/local_merge_20261117'
FINGERPRINT = '17c6759cbaf6'

def read(path):
    return json.loads(path.read_bytes())

def digest(value):
    return hashlib.sha256(value).hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

def update(record, directory, report_path, fingerprint):
    report = read(report_path)
    parts = report.get('output_parts', [report])
    record.update(input=str(directory / Path(report['output']).name),
                  report=str(report_path), report_sha256=digest(report_path.read_bytes()),
                  production_fingerprint=fingerprint)
    record['inputs'] = [
        dict(path=str(directory / Path(part['output']).name), adler32=part['output_adler32'],
             expected_entries=part['trees']['ntuple'], provenance_sha256=part['provenance_sha256'])
        for part in parts
    ]
    if record['sample'] == 'data':
        record['input'] = record['inputs'][0]['path']

def verify(item):
    result = subprocess.check_output(['xrdfs', 'eosuser.cern.ch', 'query', 'checksum', item['path']], text=True)
    if result.split() != ['adler32', item['adler32']]:
        raise ValueError(f"Checksum mismatch for {item['path']}: {result.strip()}")

def main():
    catalog = read(CORE / 'samples.json')
    for records in catalog['eras'].values():
        for region in records:
            records[region] = []
    for old in read(OLD / 'validation_inputs.json')['records']:
        record = dict(old)
        era, region = record.pop('era'), record.pop('region')
        record.pop('id', None)
        catalog['eras'][era][region].append(record)

    lookup = {}
    for path in PROD.glob('*/merge_reports/*.json'):
        report = read(path)
        if 'trees' in report:
            output = Path(report['output'])
            region, era, *_ = output.parent.parent.name.split('_')
            lookup[era, region.lower(), output.stem] = path

    status = read(SUPPLEMENT / 'status.json')
    assert status['status'] == 'complete' and len(status['completed']) == 7
    templates = {r['sample']: r for r in catalog['eras']['2022']['wcr'] if r['group'] == 'EWK'}
    templates.update({r['sample']: r for r in catalog['eras']['2024']['zee'] if r['group'] == 'EWK'})
    for added in status['completed']:
        sample = Path(added['destination']).stem
        era = added['era']
        records = catalog['eras'][era]['wcr']
        assert all(r['sample'] != sample for r in records)
        records.append(dict(templates[sample]))
        lookup[era, 'wcr', sample] = Path(added['report'])
        assert read(Path(added['report']))['output_adler32'] == added['adler32']

    import uproot
    sample_report = read(Path(status['completed'][0]['report']))
    with uproot.open(status['completed'][0]['destination']) as root:
        fingerprint = str(root['production_fingerprint'])
    assert fingerprint.startswith(FINGERPRINT) and digest(fingerprint.encode()) == sample_report['provenance_sha256']['production_fingerprint']

    for era, regions in catalog['eras'].items():
        for region, records in regions.items():
            base = SHARE / f'{region.upper()}_{era}'
            for record in records:
                sample = record['sample']
                if sample != 'data':
                    update(record, base / 'mc', lookup[era, region, sample], fingerprint)
                elif era == '2023' and region in ('wcr', 'zcr', 'srsb'):
                    path = AUDIT / 'muon_2023_v1' / f'merge_{region}.json'
                    update(record, base / 'data', path, read(path)['production_fingerprint'])
                elif era in ('2025', '2026') and region == 'zee':
                    path = AUDIT / 'zee_rebuild_v2' / f'merge_{era}.json'
                    update(record, base / 'data', path, read(path)['production_fingerprint'])

    items = [item for regions in catalog['eras'].values() for records in regions.values()
             for record in records for item in record['inputs']]
    with ThreadPoolExecutor(max_workers=12) as pool:
        for _ in pool.map(verify, items):
            pass
    write(CORE / 'input_preflight.json', {'passed': False})
    write(CORE / 'samples.json', catalog)
    print(f'Verified {len(items)} checksums', flush=True)

if __name__ == '__main__':
    main()
