"""Freeze source, scientific input and checkpoint checksums after all jobs finish."""
from remote_bootstrap import HOST,BASE
from pathlib import Path
import sys,json,platform
import numpy,scipy,pyscf
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import write,digest,plan
R=BASE/'results'
write(R/'environment.json',dict(host=HOST,python=sys.version,platform=platform.platform(),numpy=numpy.__version__,scipy=scipy.__version__,pyscf=pyscf.__version__,
 native=[dict(path=str(p),sha256=digest(p)) for p in Path('/home/enzo/nano-MATRIX-current/packages/matrix-gfn2/src/matrix_gfn2').glob('_gfn2_native*.so')]))
files=[]
for folder in [R,BASE/'prepared',BASE/'executed_sources',BASE/'prepared_B',BASE/'references_B',BASE/'holdout_B_inputs']:
 if folder.exists():files += [p for p in folder.rglob('*') if p.is_file() and p.name!='provenance.json' and '__pycache__' not in str(p)]
files+=list(BASE.glob('*.py'))+list(BASE.glob('*.log'))+list(BASE.glob('*.json'))+[BASE/'REPORT.md']
for package in ['matrix-gfn2','matrix-apoc','matrix-link']:files+=list((BASE/'source/packages'/package/'src').rglob('*.py'))
files+=list((BASE/'source/qualification').glob('test_*.py'))
files=[p for p in files if p.is_file() and not p.name.startswith('._')]
external={};cases=json.loads((BASE/'protocol.json').read_text())['cases']+[c for c in plan() if c['name'] in ['nitroxyl','uracil']]
for c in cases:
 folder=Path(c['reference']);candidates=[folder/p for p in ['reference.npz','backend.chk','baseline.npz','selected_full.npz','backend_tda.npz','result.json']]
 if 'metadata' in c:candidates.append(Path(c['metadata']))
 candidates += [Path(t['checkpoint']) for t in c['targets'] if 'checkpoint' in t]
 for p in candidates:
  if p.is_file():external[str(p)]=dict(size=p.stat().st_size,sha256=digest(p))
write(R/'provenance.json',dict(host=HOST,protocol_sha256=digest(BASE/'protocol.json'),
 artifacts={str(p.relative_to(BASE)):dict(size=p.stat().st_size,sha256=digest(p)) for p in sorted(set(files))},external_inputs=external,
 no_energy_offset=True,default_changed=False,full_ao_response_scope='vertical-only; fresh three-carbonyl holdout reported separately',
 checkpoint_policy='completed SCF references, state and density-response checkpoints retained; interrupted failed-development branches retained'))
print('HASHED',len(set(files)))
