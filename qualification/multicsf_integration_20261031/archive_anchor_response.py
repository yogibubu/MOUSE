"""Archive new anchor/second-molecule derivative evidence without old reruns."""
from pathlib import Path
import hashlib,json,socket,tarfile
assert socket.getfqdn()=='oracle.sns.it'
W=Path('/home/enzo/MOUSE-orbital-performance-20261031/multicsf_integration')
files=[];remote_only={}
for folder in ['anchor_rank','anchor_rank_seeded','anchor_rank_targeted','cpd_hessian','rigid_body']:
    for p in (W/folder).rglob('*'):
        if not p.is_file():continue
        if p.suffix=='.chk':
            remote_only[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        elif p.suffix in {'.json','.npz','.log'}:files.append(p)
for name in ['anchor_rank.py','qualify_cpd_hessian.py','anchor_rank.log','cpd_hessian.log','audit_anchor_projection.py','audit_anchor_projection.log','anchor_rank_seeded.py','anchor_rank_seeded.log','anchor_rank_targeted.py','anchor_rank_targeted.log','rigid_body_audit.py','audit_anchor_projection_incomplete_unseeded.log']:
    p=W/name
    if p.exists():files.append(p)
manifest=dict(host=socket.getfqdn(),sha256={str(p.relative_to(W)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              remote_only_sha256=remote_only,complete_anchor_test=json.loads((W/'anchor_rank_targeted/summary.json').read_text()).get('complete',False),
              complete_cpd_hessian=json.loads((W/'cpd_hessian/report.json').read_text()).get('complete',False))
path=W/'ANCHOR_RESPONSE_TRANSFER.json';path.write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(W/'anchor_response.tar.gz','w:gz') as tar:
    for p in files+[path]:tar.add(p,arcname=str(p.relative_to(W)))
print('Archived',len(files),'new artifacts; SCF backend checkpoints retained remotely',len(remote_only))
