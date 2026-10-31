"""Resume the unfinished reference response with an adequate dependence cutoff."""
from pathlib import Path
import socket,os,signal,shutil,json,hashlib,zipfile
assert socket.getfqdn()=='oracle.sns.it'
W=Path('/home/enzo/MOUSE-orbital-performance-20261031/multicsf_integration')
target=str(W/'anchor_rank_targeted.py').encode();matches=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:cmd=(p/'cmdline').read_bytes().split(b'\0')
    except (PermissionError,FileNotFoundError,ProcessLookupError):continue
    if len(cmd)>=3 and cmd[0].endswith(b'python') and cmd[1]==b'-u' and cmd[2]==target:matches.append(int(p.name))
assert len(matches)==1, matches
os.kill(matches[0],signal.SIGTERM)
provenance={}
for source in (W/'anchor_rank_targeted').glob('*/*'):
    if not source.is_dir():continue
    dest=W/'anchor_rank_converged'/source.relative_to(W/'anchor_rank_targeted');dest.mkdir(parents=True,exist_ok=True)
    for name in ['reference.npz','backend.chk','tda.npz','native.npz','native_certificate.json','ritz_checkpoint.npz']:
        p=source/name
        if not p.exists():continue
        if p.suffix=='.npz':
            try:
                with zipfile.ZipFile(p) as z:assert z.testzip() is None
            except (zipfile.BadZipFile,AssertionError):continue
        shutil.copy2(p,dest/name)
        provenance[str(dest/name)]=dict(source=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
record=dict(reason='Default lindep=1e-12 discards small preconditioned requested-root corrections before reaching a 1e-7 residual; retain them with lindep=1e-20 and resume saved Ritz vectors',
            pid=matches[0],completed_ground_preserved=True,ritz_checkpoint_preserved=True,
            electronic_operator_unchanged=True,root_residual_tolerance_unchanged=True)
(W/'anchor_rank_targeted/solver_amendment.json').write_text(json.dumps(record,indent=2)+'\n')
(W/'anchor_rank_converged/REUSED_CHECKPOINTS.json').write_text(json.dumps(provenance,indent=2)+'\n')
print('Retained',len(provenance),'completed/iterative checkpoints')
