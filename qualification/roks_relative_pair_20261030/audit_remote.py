"""Read-only Oracle source/runtime/checkpoint audit; no electronic calculations."""
import hashlib,json,socket,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('/home/enzo/MOUSE-roks-relative-pair-20261030')
assert socket.getfqdn()=='oracle.sns.it'
protocol=json.loads((ROOT/'FINAL_PROTOCOL.json').read_text());source={}
for relative,expected in protocol['source_sha256'].items():
    path=ROOT/'source'/relative
    if relative=='examples/roks_relative_pair.py':path=ROOT/'source/qualification/example_roks_relative_pair.py'
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    source[relative]=dict(expected=expected,actual=actual,matched=actual==expected)
assert all(r['matched'] for r in source.values())
native=next(Path('/home/enzo/nano-MATRIX-current/packages/matrix-gfn2/src/matrix_gfn2').glob('_gfn2_native*.so'))
report=dict(host=socket.getfqdn(),timestamp_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,
            all_source_hashes_match=True,sources=source,native_extension=dict(path=str(native),sha256=hashlib.sha256(native.read_bytes()).hexdigest()))
(ROOT/'SOURCE_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
artifacts={}
for folder in ['molecular','results','vertical','validation_G','example_smoke']:
    for path in sorted((ROOT/folder).rglob('*')):
        if path.is_file() and path.name not in {'projected_eri.npz','quadrature_progress.npz'}:
            artifacts[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
for path in ROOT.iterdir():
    if path.is_file() and path.suffix in {'.log','.json'} and path.name!='ARTIFACT_HASHES.json':
        artifacts[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
(ROOT/'ARTIFACT_HASHES.json').write_text(json.dumps(artifacts,indent=2)+'\n')
print(json.dumps(dict(host=report['host'],matched_source_files=len(source),artifact_files=len(artifacts))))
