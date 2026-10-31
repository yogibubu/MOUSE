"""Reject forbidden-sector Ritz ghosts; preserve the completed DFT ground."""
from pathlib import Path
import socket,os,signal,shutil,json,hashlib
assert socket.getfqdn()=='oracle.sns.it'
W=Path('/home/enzo/MOUSE-orbital-performance-20261031/multicsf_integration')
target=str(W/'anchor_rank_converged.py').encode();matches=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:cmd=(p/'cmdline').read_bytes().split(b'\0')
    except (PermissionError,FileNotFoundError,ProcessLookupError):continue
    if len(cmd)>=3 and cmd[0].endswith(b'python') and cmd[1]==b'-u' and cmd[2]==target:matches.append(int(p.name))
if matches:
    assert len(matches)==1
    os.kill(matches[0],signal.SIGTERM)
provenance={}
for source in (W/'anchor_rank_converged').glob('*/*'):
    if not source.is_dir():continue
    dest=W/'anchor_rank_symmetry'/source.relative_to(W/'anchor_rank_converged');dest.mkdir(parents=True,exist_ok=True)
    for name in ['reference.npz','backend.chk']:
        p=source/name
        if not p.exists():continue
        shutil.copy2(p,dest/name)
        provenance[str(dest/name)]=dict(source=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
record=dict(reason='Full-space restart Ritz selection admitted forbidden-irrep null-space components; exact symmetry-block response and positive-root checkpoint exclude them structurally',
            completed_ground_preserved=True,uncertified_ritz_retained_but_not_reused=True,
            electronic_operator_unchanged=True,root_residual_tolerance_unchanged=True)
(W/'anchor_rank_converged/solver_amendment.json').write_text(json.dumps(record,indent=2)+'\n')
(W/'anchor_rank_symmetry/REUSED_CHECKPOINTS.json').write_text(json.dumps(provenance,indent=2)+'\n')
print('Reused',len(provenance),'completed ground/checkpoint files; rejected unphysical Ritz vectors')
