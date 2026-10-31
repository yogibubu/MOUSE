"""Prepare fully relaxed DFT controls; keep all completed ground references."""
from remote_bootstrap import HOST,BASE
from pathlib import Path
import json
from spin_kernel_probe import plan,OLD,write,digest
ROOT=BASE/'results/full_reference_inputs';OUT=ROOT/'frozen_relaxation';OUT.mkdir(parents=True,exist_ok=True)
cases=[]
source_cases=[(c,BASE/'results',OLD) for c in plan()]
for p in (BASE/'holdout_data').glob('*/case.json'):
 source_cases.append((json.loads(p.read_text()),BASE/'results/holdout',BASE/'holdout_data'))
for case,prefix,baseline in source_cases:
 source=json.loads((baseline/'frozen_relaxation'/case['name']/'result.json').read_text())
 target=json.loads((prefix/'bhandhlyp'/case['name']/'result.json').read_text())
 response=json.loads((prefix/'response_kernel'/case['name']/'result.json').read_text())
 rows=[];targets=[]
 for t in case['targets']:
  a=next(x for x in source['records'] if x['id']==t['id'])
  b=next(x for x in target['records'] if x['id']==t['id'])
  r=next(x for x in response['records'] if x['id']==t['id'])
  if not a.get('qualified') or not b.get('qualified') or not r.get('qualified'):continue
  row=dict(a,dft_hole=b['dft_hole'],dft_particle=b['dft_particle'],dft_frozen_ev=b['frozen_ev'],
   se_relaxation_ev=r['relaxation_ev'],hybrid_ev=b['frozen_ev']+r['relaxation_ev'],
   hybrid_error_ev=b['frozen_ev']+r['relaxation_ev']-t['tbe_ev'])
  rows.append(row);targets.append(t)
 if not targets:continue
 ground=json.loads((prefix/'bhandhlyp'/case['name']/'ground.json').read_text())
 write(OUT/case['name']/'result.json',dict(host=HOST,ground=ground,records=rows))
 cases.append(dict(case,reference=str(prefix/'bhandhlyp'/case['name']),targets=targets))
write(OUT/'plan.json',dict(host=HOST,cases=cases,formula='fully relaxed common-orbital BH&HLYP ROKS, no relaxation transfer',
 source_model_selection='BH&HLYP fixed before evaluating its relaxed-state energies',
 previous_frozen_transfer_holdout='failed; not reused as a successful validation'))
print([(c['name'],[t['id'] for t in c['targets']]) for c in cases])
