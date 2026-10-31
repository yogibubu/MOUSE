"""Verify the downloaded artifact bytes against the Oracle manifest."""
from pathlib import Path
import json,hashlib
BASE=Path(__file__).resolve().parent
ROOT=BASE/'remote'
expected=json.loads((ROOT/'ARTIFACT_HASHES.json').read_text())
for name,digest in expected.items():
    path=ROOT/name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        raise RuntimeError('artifact checksum mismatch: '+name)
source=json.loads((ROOT/'SOURCE_VERIFICATION.json').read_text())
protocol=json.loads((BASE/'PROTOCOL.json').read_text())
assert source['all_source_hashes_match']
assert all(r['matched'] and r['actual']==protocol['source_sha256'][name] for name,r in source['sources'].items())
example=json.loads((ROOT/'example_validation.json').read_text())
assert example['passed'] and example['restarted'] and example['energy_change_hartree']<1e-10
suite=(ROOT/'tests_full_seeded.log').read_text().strip().splitlines()[-1]
assert suite.startswith('86 passed'),suite
report=dict(verified_files=len(expected),verified_source_files=len(source['sources']),
    all_artifact_hashes_match=True,all_source_hashes_match=True,example_restart=example,remote_full_tests=suite)
(BASE/'TRANSFER_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
