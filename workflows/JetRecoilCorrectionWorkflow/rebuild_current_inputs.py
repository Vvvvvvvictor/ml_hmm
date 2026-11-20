#!/usr/bin/env python3
"""Rebuild the nominal catalog from the accepted 641e production reports."""
import hashlib
import json
from pathlib import Path

CORE = Path(__file__).resolve().parent
BASE = Path('/afs/cern.ch/user/j/jiehan/private/CROWN_run3/logs/medmed_no_btag_20261110_runtimefix')
FINGERPRINT = '641e290e583fe6415f7bda278589601d6251786adb290eb46df2dac5fdf28045'


def main():
    mapping = json.loads((CORE / 'input_directory_mapping_641e290e583f.json').read_text())
    def mapped(path):
        path = str(path)
        for old, new in mapping.items():
            if path == old or path.startswith(old + '/'):
                return new + path[len(old):]
        raise ValueError(f'Output does not belong to current production: {path}')
    lookup = {}
    for batch in ('2223', '2426'):
        acceptance = json.loads((BASE / batch / 'acceptance.json').read_text())
        if acceptance['status'] != 'passed' or acceptance['production']['fingerprint'] != FINGERPRINT:
            raise ValueError(f'Production not accepted: {batch}')
        for report in (BASE / batch / 'merge_reports').glob('*.json'):
            raw = report.read_bytes()
            data = json.loads(raw)
            output = Path(mapped(data['output']))
            folder = output.parent if output.parent.name.startswith(('SRSB_', 'ZCR_', 'ZEE_', 'WCR_')) else output.parent.parent
            region, era = folder.name.split('_')
            sample = 'data' if output.name == 'data.root' else output.stem
            key = (era, region.lower(), sample)
            if key in lookup: raise ValueError(f'Duplicate report: {key}')
            lookup[key] = (report, hashlib.sha256(raw).hexdigest(), data)
    catalog = json.loads((CORE / 'samples.json').read_text())
    eras = json.loads((CORE / 'eras.json').read_text())
    count = 0
    for era, regions in catalog['eras'].items():
        eras[era]['production_fingerprint'] = FINGERPRINT
        for region, records in regions.items():
            eras[era]['inputs'][region] = str(CORE.parent / f'{region.upper()}_{era}')
            for record in records:
                report, sha, data = lookup[(era, region, record['sample'])]
                parts = data.get('output_parts', [data])
                inputs = []
                for part in parts:
                    path = Path(mapped(part['output']))
                    if not path.is_file(): raise FileNotFoundError(path)
                    inputs.append(dict(path=str(path), adler32=part['output_adler32'], expected_entries=part['trees']['ntuple'], provenance_sha256=part['provenance_sha256']))
                    count += 1
                record.update(input=mapped(data['output']), inputs=inputs, production_fingerprint=FINGERPRINT, report=str(report), report_sha256=sha)
    (CORE / 'input_preflight.json').write_text(json.dumps({'passed': False, 'reason': 'new catalog requires preflight'}, indent=2)+'\n')
    (CORE / 'eras.json').write_text(json.dumps(eras, indent=2)+'\n')
    (CORE / 'samples.json').write_text(json.dumps(catalog, indent=2)+'\n')
    print(f'Rebuilt {count} file records; preflight required')

if __name__ == '__main__': main()
