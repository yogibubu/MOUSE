"""Compare fixed combinations, keeping arithmetic averaging and state branches explicit."""
from remote_bootstrap import HOST,BASE
from pathlib import Path
import json,sys,numpy as np
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import write,digest
from combined_response import CASES,LABELS,ROOT
R=BASE/'results';ab=json.loads((R/'response_ablation/summary.json').read_text())['rows'];rows=[]
for case in CASES:
 name=case['name']
 for t in case['targets']:
  state=t['id'];models={}
  for label,folder in [('raw_full',R/'full_response'),('spectral',R/'qualified_spectral_response')]+[(x,ROOT/x) for x in LABELS]:
   path=folder/name/state/'result.json'
   models[label]=json.loads(path.read_text()) if path.exists() else dict(qualified=False,status='not calculated after development rejection')
  raw=models['raw_full'];spectral=models['spectral'];mean=.5*(raw.get('excitation_ev',float('nan'))+spectral.get('excitation_ev',float('nan')))
  models['energy_mean_50_50']=dict(qualified=bool(raw.get('qualified') and spectral.get('qualified')),excitation_ev=mean,error_ev=mean-t['tbe_ev'],scope='arithmetic control only; not a variational stationary state')
  native=next((x['variants']['native_control'] for x in ab if x['molecule']==name and x['state']==state),None)
  if native is None:
   held=json.loads((R/'holdout_C'/name/'result.json').read_text());native=dict(qualified=held.get('qualified'),excitation_ev=held['native_roks_ev'],error_ev=held['native_roks_ev']-held['tbe_ev'])
  models['native']=native
  group='development' if name in ['nitroxyl','uracil'] else ('B_exposed' if name in ['acetaldehyde','acetone','formamide'] else 'C_exposed_for_combinations')
  record=dict(molecule=name,state=state,tbe_ev=t['tbe_ev'],group=group,models=models,tda={})
  for label in LABELS:
   path=ROOT/label/name/'tda.json'
   if path.exists():record['tda'][label]=json.loads(path.read_text())
  if name=='uracil':
   warm=R/'qualified_spectral_continuation'/state/'t_0.50'
   alternatives=[]
   for p in warm.glob('index_*/result.json'):
    rr=json.loads(p.read_text())
    if rr.get('qualified'):alternatives.append(rr)
   record['mixed_functional_warm_alternatives']=alternatives
   cold=models['mixed_functional_50_50'];record['mixed_functional_branch_ambiguity']=bool(cold.get('qualified') and any(abs(x['excitation_ev']-cold['excitation_ev'])>1e-4 for x in alternatives))
  rows.append(record)
labels=['native','raw_full','spectral','energy_mean_50_50']+LABELS
stats={}
for group,select in [('all',lambda r:True),('n_pi',lambda r:r['state']=='n_pi'),('development',lambda r:r['group']=='development'),('B_exposed',lambda r:r['group']=='B_exposed'),('C_exposed',lambda r:r['group']=='C_exposed_for_combinations')]:
 paired=[r for r in rows if select(r) and all(r['models'][l].get('qualified') for l in labels if l!='matched_diagonal_full_couplings')]
 stats[group]={}
 for label in labels:
  errors=[r['models'][label]['error_ev'] for r in paired if r['models'][label].get('qualified')]
  stats[group][label]=dict(count=len(errors),mae_ev=float(np.mean(np.abs(errors))) if errors else None,maxabs_ev=max(map(abs,errors),default=None))
write(R/'combined_summary.json',dict(host=HOST,protocol_sha256=digest(ROOT/'protocol.json'),rows=rows,statistics=stats,no_fitted_weights=True,no_state_energy_offset=True,scope='subsequent exposed-data comparison, not blind validation'))
print('COMBINATION SUMMARY',stats,flush=True)
