from remote_bootstrap import HOST,BASE
from pathlib import Path
import hashlib,json,sys,platform
import numpy,scipy,pyscf
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results'
# Clarify the scope of the recorded qualification flag without repeating spectra.
for path in [*OUT.glob('comparison.json'),*OUT.glob('*_comparison.json'),*OUT.glob('scale_*/*comparison.json')]:
 d=json.loads(path.read_text());records=d.get('references',[d])
 for record in records:
  for row in record['tda']:
   if 'qualified' in row:row['state_mapping_qualified']=row.pop('qualified')
 path.write_text(json.dumps(d,indent=2)+'\n')
files=[BASE/'source/packages/matrix-gfn2/src/matrix_gfn2'/n for n in ['hybrid_spectrum.py','__init__.py','response.py','deltascf.py','roks.py']]
files += [BASE/'source/qualification'/n for n in ['test_hybrid_spectrum.py','test_roks.py','test_deltascf.py','test_deltascf_link.py','test_orbital_saddle.py']]
refs=Path('/home/enzo/MATRIX-runs/tda-uracil-application-20261022-v1/docs/reports/tda_uracil_application_20261022/results')
holdout=Path('/home/enzo/MATRIX-runs/tda-frozen-validation-20261022-v1/docs/reports/tda_frozen_validation_20261022/thymine')
d=dict(host=HOST,python=sys.version,numpy=numpy.__version__,scipy=scipy.__version__,pyscf=pyscf.__version__,platform=platform.platform(),thread_limits=2,uracil_new_scf_jobs=0,thymine_new_ground_reference_jobs=1,thymine_new_roks_states=2,thymine_resume_policy='ground and converged excited checkpoints reused; incomplete seed metadata repaired without restarting completed orbitals',source_sha256={str(p.relative_to(BASE/'source')):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},scripts_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},reference_npz_sha256={name:hashlib.sha256((refs/name/'reference.npz').read_bytes()).hexdigest() for name in ['b3lyp_avdz','pbe0_avdz','pbe0_avtz']},holdout_reference_sha256={name:hashlib.sha256((holdout/'results'/name/'reference.npz').read_bytes()).hexdigest() for name in ['pbe0_avdz','pbe0_avtz']},artifact_sha256={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file()})
(ROOT/'provenance.json').write_text(json.dumps(d,indent=2)+'\n')
