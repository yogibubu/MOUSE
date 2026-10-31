"""Freeze actual artifact checksums after computation; no numerical jobs."""
from remote_bootstrap import HOST,BASE
from pathlib import Path
import json,sys,platform,subprocess
import numpy,scipy,pyscf
from spin_kernel_probe import write,digest,OLD,plan
R=BASE/'results'
write(R/'environment.json',dict(host=HOST,python=sys.version,platform=platform.platform(),
 numpy=numpy.__version__,scipy=scipy.__version__,pyscf=pyscf.__version__,
 numerical_host_policy='Oracle only; laptop orchestration and file inspection',
 native_binary=[dict(path=str(p),sha256=digest(p)) for p in Path('/home/enzo/nano-MATRIX-current/packages/matrix-gfn2/src/matrix_gfn2').glob('_gfn2_native*.so')]))
# The running group jobs loaded the archived original-engine script. The source
# on disk was later repaired for rectangular MO coefficient matrices. Result
# script_sha256 is a finish-time disk hash, not always the loaded-code hash.
execution={}
for p in (R/'full_reference').glob('*/*/result.json'):
 d=json.loads(p.read_text());mol=d['molecule'];state=d['state']
 if 'nmo' in d:source=BASE/'full_reference.py';engine=BASE/'full_reference_engine.py'
 elif mol in ['nitroxyl','nitrous_acid'] or (mol in ['uracil','thymine'] and state=='n_pi'):
  source=BASE/'executed_sources/full_reference_before_freeze_release.py';engine=OLD/'dft_relaxation.py'
 else:source=BASE/'executed_sources/full_reference_freeze_release_original_engine.py';engine=OLD/'dft_relaxation.py'
 execution[f'{mol}/{state}']=dict(loaded_source=str(source),loaded_source_sha256=digest(source),
  engine=str(engine),engine_sha256=digest(engine),finish_time_disk_sha256=d.get('script_sha256'),
  note='finish-time source hash can differ from code loaded by an already running process; archived code is authoritative')
write(R/'executed_source_mapping.json',execution)
files=[]
for folder in [R,BASE/'holdout_data',BASE/'executed_sources']:
 for p in folder.rglob('*'):
  if p.is_file() and p.name!='provenance.json' and '__pycache__' not in str(p):files.append(p)
files+=list(BASE.glob('*.py'))+list(BASE.glob('*.log'))+[BASE/'REPORT.md']
for package in ['matrix-gfn2','matrix-apoc','matrix-link']:
 files+=list((BASE/'source/packages'/package/'src').rglob('*.py'))
files+=list((BASE/'source/qualification').glob('test_*.py'))
files=[p for p in files if not p.name.startswith('._')] # macOS AppleDouble metadata is not source code
external={}
cases=plan()+[json.loads(p.read_text()) for p in (BASE/'holdout_data').glob('*/case.json')]
for c in cases:
 candidates=[Path(c['reference'])/name for name in ['reference.npz','backend.chk']]
 candidates+=[Path(t['checkpoint']) for t in c['targets'] if 'checkpoint' in t]
 baseline=(BASE/'holdout_data' if (BASE/'holdout_data'/c['name']/'case.json').exists() else OLD)/'frozen_relaxation'/c['name']/'result.json'
 candidates.append(baseline)
 for p in candidates:
  if p.is_file():external[str(p)]=dict(size=p.stat().st_size,sha256=digest(p))
write(R/'provenance.json',dict(host=HOST,external_inputs=external,artifacts={str(p.relative_to(BASE)):dict(size=p.stat().st_size,sha256=digest(p)) for p in sorted(set(files))},
 source_mapping='executed_source_mapping.json',checkpoint_policy='completed qualified results and response columns reused; alternate optimizer attempts preserved',
 holdout_protocol_sha256=digest(R/'holdout_protocol.json')))
print('HASHED',len(set(files)))
