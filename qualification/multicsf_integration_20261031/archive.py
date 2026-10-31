"""Snapshot the completed qualification, including portable furan inputs."""
from pathlib import Path
import json,hashlib,tarfile,socket
assert socket.getfqdn()=='oracle.sns.it'
V=Path('/home/enzo/MOUSE-orbital-performance-20261031');W=V/'multicsf_integration';files=[];remote_only={}
for p in W.rglob('*'):
 if not p.is_file() or 'source' in p.relative_to(W).parts or '__pycache__' in p.parts:continue
 rel=p.relative_to(W)
 if p.suffix not in {'.json','.npz','.xml','.log','.py'}:continue
 if p.suffix=='.npz' and ('B022_pre_conversion_fix' in p.parts or 'relocated_model' in p.parts or ('Phthalimide' in p.parts and p.stat().st_size>2_000_000)):
  remote_only[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();continue
 files.append(p)
for p in (V/'pi_pi_repair/nuclear_validation').glob('*'):
 if p.name.startswith('full_columns_') or p.name=='Furan_full_columns_report.json':files.append(p)
files.append(V/'test_multicsf_all_columns.log')
# Record every source in the three core packages; transfer only new/changed ones.
source_sha={}
for package in ['matrix-gfn2','matrix-apoc','matrix-link']:
 for p in (W/'source/packages'/package/'src').rglob('*.py'):source_sha[str(p.relative_to(W/'source'))]=hashlib.sha256(p.read_bytes()).hexdigest()
native=next(Path('/home/enzo/nano-MATRIX-current/packages/matrix-gfn2/src/matrix_gfn2').glob('_gfn2_native*.so'))
manifest=dict(host=socket.getfqdn(),sha256={str(p.relative_to(V)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},source_sha256=source_sha,remote_only=remote_only,native_extension=dict(path=str(native),sha256=hashlib.sha256(native.read_bytes()).hexdigest()))
path=W/'TRANSFER.json';path.write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(W/'qualification.tar.gz','w:gz') as tar:
 for p in files+[path]:tar.add(p,arcname=str(p.relative_to(V)))
print('Archived',len(files),'files; large/duplicate inputs retained remotely',len(remote_only))
