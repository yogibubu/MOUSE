"""Archive inputs actually read by the symmetry-only external comparison."""
from pathlib import Path
import json,hashlib,socket,tarfile
assert socket.getfqdn()=='oracle.sns.it'
P=Path('/home/enzo/MOUSE-publication-validation-20261030')
O=P/'external/dftbplus24/symmetry_audit_20261031'
td=json.loads((O/'report.json').read_text());de=json.loads((O/'delta_report.json').read_text())
specs=json.loads((P/'inputs/target_references.json').read_text());files=set()
for r in td['rows']:
    if 'source' not in r:continue
    folder=Path(r['source'])
    for name in ['geometry.xyz','eigenvec.out','TRA.DAT','dftb_in.hsd','result.json']:
        if (folder/name).exists():files.add(folder/name)
    ref=Path(specs[r['molecule']]['reference'])
    files.update([ref/'reference.npz',ref/'backend.chk'])
    files.add(next(p for p in [ref/'selected_full.npz',ref/'backend_tda.npz',ref/'baseline.npz'] if p.exists()))
for r in de['rows']:
    if 'source' not in r:continue
    folder=Path(r['source'])
    for name in ['geometry.xyz','eigenvec.out','results.tag','detailed.out','dftb_in.hsd','result.json']:
        if (folder/name).exists():files.add(folder/name)
files.update(p for p in O.glob('*.json') if p.name!='TRANSFER.json')
files.update([P/'assign_dftb_symmetry.py',P/'assign_delta_dftb_symmetry.py'])
files.update([P/'inputs/target_references.json',P/'external/dftbplus24/attempt_03/protocol.json'])
def arc(p):
    try:return 'campaign/'+str(p.relative_to(P))
    except ValueError:return 'inherited_reference/'+str(p).lstrip('/')
manifest=dict(host=socket.getfqdn(),scope='all inputs actually read by TD/Delta symmetry audits; original SCF jobs not repeated',
              sha256={arc(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},
              sources={arc(p):str(p) for p in sorted(files)})
names=[arc(p) for p in files];assert len(set(names))==len(files)
path=O/'TRANSFER.json';path.write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(O/'evidence.tar.gz','w:gz',dereference=True) as tar:
    for p in sorted(files):tar.add(p,arcname=arc(p))
    tar.add(path,arcname='TRANSFER.json')
print('Archived',len(files),'input/result artifacts',sum(p.stat().st_size for p in files),'bytes')
