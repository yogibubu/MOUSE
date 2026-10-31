"""Collect immutable scientific provenance after frozen-energy checks finish."""
from remote_bootstrap import HOST,BASE,native_path
from pathlib import Path
import hashlib,json,platform,shutil,sys
import numpy,scipy,pyscf
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'frozen_relaxation'
plan=json.loads((OUT/'plan.json').read_text());inputs={}
for case in plan['cases']:
 folder=Path(case['reference'])
 for p in [folder/'reference.npz',folder/'backend.chk',*[Path(t['checkpoint']) for t in case['targets'] if 'checkpoint' in t]]:
  inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
meta={
 'Uracil.json':ROOT.parent/'uracil_uridine_20261029/inputs/QUEST_Uracil.json',
 'Thymine.json':Path('/home/enzo/MATRIX-runs/tda-frozen-validation-20261022-v1/docs/reports/tda_frozen_validation_20261022/thymine/inputs/Thymine.json'),
 **{name:ROOT/'generalization/inputs'/name for name in ['Nitroxyl.json','Nitrous_acid.json','Propynal.json','Pyridine.json','Cytosine.json']},
 'QUEST_LICENSE':ROOT/'generalization/inputs/QUEST_LICENSE'}
(OUT/'inputs').mkdir(exist_ok=True)
for name,p in meta.items():shutil.copyfile(p,OUT/'inputs'/name);inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
files=[ROOT/n for n in ['frozen_relaxation.py','frozen_grid_check.py','frozen_relaxation_summary.py','frozen_relaxation_audit.py','frozen_relaxation_provenance.py','remote_bootstrap.py','extent_support.py','generalization.py']]
files+=[BASE/'source/packages/matrix-gfn2/src/matrix_gfn2'/n for n in ['calculator.py','deltascf.py','roks.py','deltascf_derivatives.py','spin_polarization.py','hybrid_spectrum.py','data/spin_constants.toml']]
files+=[native_path,OUT/'protocol.json']
provenance=dict(host=HOST,python=sys.version,platform=platform.platform(),numpy=numpy.__version__,scipy=scipy.__version__,pyscf=pyscf.__version__,threads_per_job=2,max_memory_mb_per_job=6000,new_scf_optimizations=0,new_excited_optimizations=0,sources_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},inputs_sha256=inputs,artifact_sha256={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='provenance.json' and p.suffix!='.log'})
(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print('Recorded',len(inputs),'input hashes and',len(provenance['artifact_sha256']),'scientific artifacts.')
