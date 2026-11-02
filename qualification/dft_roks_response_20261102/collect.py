"""Export source-linked analytic-response certificates from the compute host."""
from pathlib import Path
import socket,json,hashlib,shutil,platform
import numpy as np
import scipy,pyscf
assert socket.getfqdn().lower()=='mac-2.homenet.telecomitalia.it'
r=Path(__file__).resolve().parent;out=r/'evidence';out.mkdir(exist_ok=True)
cache=json.loads((r/'cached_response_validation.json').read_text());assert len(cache['cases'])==6
assert all(row['complete_Hessian_error']<1e-8 for row in cache['cases'])
source=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/source/packages/matrix-pyscf/src/matrix_pyscf')
for p in r.iterdir():
 if p.is_file() and p.suffix in ('.json','.npz','.py'):shutil.copy2(p,out/p.name)
for p in r.glob('*.log'):
 target=out/'logs'/(p.stem+'.txt');target.parent.mkdir(exist_ok=True);shutil.copy2(p,target)
for p in source.glob('*.py'):
 target=out/'executed_sources'/p.name;target.parent.mkdir(exist_ok=True);shutil.copy2(p,target)
for p in (r/'executed_sources_initial').glob('*.py'):
 target=out/'executed_sources_initial'/p.name;target.parent.mkdir(exist_ok=True);shutil.copy2(p,target)
records=[]
for p in out.glob('*_validation.json'):
 record=json.loads(p.read_text())
 if 'method' in record:records.append(record)
assert len(records)==6
summary=dict(host=socket.getfqdn(),Python=platform.python_version(),PySCF=pyscf.__version__,SciPy=scipy.__version__,NumPy=np.__version__,
 independent_cases=6,ROKS_cases=5,unrestricted_DeltaSCF_cases=1,molecules=2,finite_differences_in_production=False,
 maximum_DFT_Hessian_vector_error=max(max(x['Hessian_vector_error'] for x in row['checks']) if 'checks' in row else row['Hessian_vector_error'] for row in records if row['method']!='HF'),
 maximum_complete_cache_Hessian_error=max(row['complete_Hessian_error'] for row in cache['cases']),
 maximum_translation_sum_rule_error=max(row['translation_sum_rule_error'] for row in cache['cases']),
 spin_scope='ROKS CSF is spin pure; functional energy is approximate common-orbital multiplet sum; DeltaSCF determinant remains spin contaminated')
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
manifest={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'}
(out/'artifact_manifest.json').write_text(json.dumps(dict(host=socket.getfqdn(),sha256=manifest),indent=2)+'\n');print(json.dumps(summary,indent=2))
