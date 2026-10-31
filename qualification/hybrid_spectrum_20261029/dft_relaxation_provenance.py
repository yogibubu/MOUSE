"""Hash the diagnostic inputs, scripts, checkpoints and reports on Oracle."""
from remote_bootstrap import HOST
from pathlib import Path
from datetime import datetime,timezone
import json
from frozen_relaxation import digest,write
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'dft_relaxation'
plan=json.loads((ROOT/'frozen_relaxation/plan.json').read_text());inputs={}
for case in plan['cases']:
 if case['name'] not in ['uracil','thymine']:continue
 folder=Path(case['reference'])
 for path in [folder/'reference.npz',folder/'backend.chk',ROOT/'frozen_relaxation'/case['name']/'result.json']:
  inputs[str(path)]=digest(path)
 for target in case['targets']:
  path=Path(target['checkpoint']);inputs[str(path)]=digest(path)
scripts={p.name:digest(p) for p in ROOT.glob('dft_*.py')}
artifacts={str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*')
 if p.is_file() and p.name!='provenance.json' and p.suffix in ['.json','.md','.npz']}
write(OUT/'provenance.json',dict(host=HOST,created_utc=datetime.now(timezone.utc).isoformat(),
 inputs=inputs,scripts=scripts,artifacts=artifacts))
print('hashed',len(inputs),'inputs,',len(scripts),'scripts,',len(artifacts),'artifacts')
