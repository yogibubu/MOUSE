"""Archive new anchor/second-molecule derivative evidence without old reruns."""
from pathlib import Path
import hashlib,json,socket,tarfile,os,platform,shutil
assert socket.getfqdn()=='oracle.sns.it'
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy,pyscf
W=Path('/home/enzo/MOUSE-orbital-performance-20261031/multicsf_integration')
support=Path('/home/enzo/MOUSE-deltascf-validation-20261029/hybrid_spectrum_20261029/extent_support.py')
(W/'support').mkdir(exist_ok=True);shutil.copy2(support,W/'support/extent_support.py')
files=[];remote_only={}
for folder in ['anchor_rank','anchor_rank_seeded','anchor_rank_targeted','anchor_rank_converged','anchor_rank_symmetry','cpd_hessian','rigid_body','support']:
    for p in (W/folder).rglob('*'):
        if not p.is_file():continue
        if p.suffix=='.chk':
            remote_only[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        elif p.suffix in {'.json','.npz','.log','.py'}:files.append(p)
for name in ['anchor_rank.py','qualify_cpd_hessian.py','anchor_rank.log','cpd_hessian.log','audit_anchor_projection.py','audit_anchor_projection.log','anchor_rank_seeded.py','anchor_rank_seeded.log','anchor_rank_targeted.py','anchor_rank_targeted.log','anchor_rank_converged.py','anchor_rank_converged.log','anchor_rank_symmetry.py','anchor_rank_symmetry.log','anchor_rank_symmetry_logger_setup.log','resume_anchor_symmetry.py','fix_response_logger.py','resume_anchor_ritz.py','audit_anchor_projection_converged.log','rigid_body.log','rigid_body_audit.py','audit_anchor_projection_incomplete_unseeded.log']:
    p=W/name
    if p.exists():files.append(p)
native=next(Path('/home/enzo/nano-MATRIX-current/packages/matrix-gfn2/src/matrix_gfn2').glob('_gfn2_native*.so'))
native_sha=hashlib.sha256(native.read_bytes()).hexdigest()
original=json.loads((W/'TRANSFER.json').read_text())['native_extension']['sha256']
assert native_sha==original, 'shared native extension changed since integration qualification'
manifest=dict(host=socket.getfqdn(),environment=dict(python=platform.python_version(),numpy=numpy.__version__,pyscf=pyscf.__version__),
              native_extension_sha256=native_sha,native_extension_matches_original_integration=True,
              sha256={str(p.relative_to(W)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              remote_only_sha256=remote_only,complete_anchor_test=json.loads((W/'anchor_rank_symmetry/summary.json').read_text()).get('complete',False),
              complete_cpd_hessian=json.loads((W/'cpd_hessian/report.json').read_text()).get('complete',False))
path=W/'ANCHOR_RESPONSE_TRANSFER.json';path.write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(W/'anchor_response.tar.gz','w:gz') as tar:
    for p in files+[path]:tar.add(p,arcname=str(p.relative_to(W)))
print('Archived',len(files),'new artifacts; SCF backend checkpoints retained remotely',len(remote_only))
