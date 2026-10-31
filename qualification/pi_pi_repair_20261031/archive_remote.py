"""Collect this experiment only, with hashes; omit large screening tensors."""
from pathlib import Path
import json,hashlib,tarfile
V=Path('/home/enzo/MOUSE-orbital-performance-20261031');out=V/'pi_pi_repair';files=[]
for p in out.rglob('*'):
    rel=p.relative_to(V)
    if p.is_file() and p.suffix in {'.json','.npz','.xml','.log'} and 'native_screened_models' not in p.parts and 'source' not in p.parts and '__pycache__' not in p.parts:files.append(p)
for pattern in ['test_multicsf*.log','audit_multicsf.log','diagnose_native_screening.log','*multicsf*.py','multicsf*.py','diagnose_pair_rank.py','diagnose_csf_coupling.py','diagnose_native_screening.py','parent_geometry_model.py','select_effective_rank.py','effective_rank_protocol.json']:
    files.extend(V.glob(pattern))
files=sorted(set(files));hashes={str(p.relative_to(V)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
# Record upstream model inputs without copying large DFT ERIs.
D=V.parent/'MOUSE-state-following-20261031/reduced_validation';P=V.parent/'MOUSE-publication-validation-20261030'
upstream={}
for name in ['Furan','Cyclopentadiene','Phthalimide','Adenine','Aza-naphthalene','Benzoquinone','Bimane_Syn']:
    paths=[D/'prepared'/name/'ground.npz',D/'models'/name/'model.npz',P/'results/full_response'/name/'model.npz',P/'results/full_response'/name/'projected_eri.npz']+[D/'tda'/name/str(i)/'spectrum.npz' for i in range(2)]
    for p in paths:
        if p.exists():upstream[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.suffix=='.py'}
manifest=V/'pi_pi_export_manifest.json';manifest.write_text(json.dumps(dict(host='oracle.sns.it',sha256=hashes,source_sha256=source_hashes,upstream_input_sha256=upstream,large_screening_model_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'native_screened_models').glob('*/model.npz')}),indent=2)+'\n')
with tarfile.open(V/'pi_pi_export.tar.gz','w:gz') as t:
    for p in files+[manifest]:t.add(p,arcname=str(p.relative_to(V)))
print('Archived',len(files),'files')
