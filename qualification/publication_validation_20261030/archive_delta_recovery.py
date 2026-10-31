"""Retain every bounded external-program recovery attempt and its checkpoints."""
from pathlib import Path
import json,hashlib,tarfile,socket
assert socket.getfqdn()=='oracle.sns.it'
P=Path('/home/enzo/MOUSE-publication-validation-20261030')
files=[]
for name in ['attempt_05','attempt_06']:
    root=P/'external/dftbplus24'/name
    files.extend(p for p in root.rglob('*') if p.is_file())
files.extend(P/name for name in ['recover_delta_dftb_numerics.py','recover_delta_dftb_initialcharges.py','recover_delta_dftb_numerics.log','recover_delta_dftb_initialcharges.log'])
O=P/'external/dftbplus24/recovery_audit_20261031';O.mkdir(exist_ok=True)
manifest=dict(host=socket.getfqdn(),scope='all bounded solver recoveries; no new Delta energy assigned unless converged and independently symmetry audited',
              binary_restart='attempt_05 rejected by backend before SCC: unsupported for Delta DFTB',
              atomic_initial_charges='attempt_06 resumes converged TD atomic charges via supported InitialCharges; all three mixers fail the unchanged SCC tolerance for both molecules',
              sha256={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
path=O/'TRANSFER.json';path.write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(O/'evidence.tar.gz','w:gz') as tar:
    for p in files:tar.add(p,arcname=str(p.relative_to(P)))
    tar.add(path,arcname='TRANSFER.json')
print('Archived',len(files),'bounded-recovery artifacts')
