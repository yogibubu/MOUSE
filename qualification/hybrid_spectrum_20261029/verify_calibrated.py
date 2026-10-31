"""Check held-out formula predictions against actual molecular API outputs."""
from remote_bootstrap import HOST
from pathlib import Path
import json,numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/extent/calibration'
cal=json.loads((OUT/'comparison.json').read_text());profile=json.loads((OUT/'profile_pbe0_uracil.json').read_text())
scale=profile['reference_energy_scale'];power=profile['extent_power'];tag=f'{scale:.12g}'
primary=json.loads((ROOT/f'results/scale_{tag}_occupied_extent_p{power:.12g}/comparison.json').read_text())
hold=json.loads((ROOT/f'results/extent/thymine/comparison_scale_{tag}_p{power:.12g}.json').read_text())
rows=[]
for name in ['pbe0_avdz','pbe0_avtz']:
 predicted=next(x for x in cal['records'] if x['reference']==name and x['policy']=='fixed_power_1.5_one_scale')
 uro=next(x for x in primary['references'] if x['reference']==name)
 assert uro['extent_power']==power and uro['reference_energy_scale']==scale
 for state,k in [('n_pi',0),('pi_pi',1)]:
  actual=next(x for x in uro['targets'] if x['state']==state)['hybrid_roks_ev'];expected=predicted['uracil']['energies_ev'][k]
  rows.append(dict(molecule='uracil',reference=name,state=state,deviation_ev=actual-expected))
  actual=next(x for x in hold['records'] if x['reference']==name and x['policy']=='basis_compactness' and x['state']==state)['roks_ev'];expected=predicted['thymine']['energies_ev'][k]
  rows.append(dict(molecule='thymine',reference=name,state=state,deviation_ev=actual-expected))
error=max(abs(x['deviation_ev']) for x in rows);assert error<1e-11
result=dict(host=HOST,molecular_values_checked=len(rows),maximum_formula_api_difference_ev=error,rows=rows)
(OUT/'molecular_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
