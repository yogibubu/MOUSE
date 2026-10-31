"""Select one fixed profile using development data, before the new holdout."""
from remote_bootstrap import HOST,BASE
from pathlib import Path
import json,hashlib
from datetime import datetime,timezone
from spin_kernel_probe import OLD,write,digest
ROOT=BASE/'results'
baseline={}
for p in (OLD/'frozen_relaxation').glob('*/result.json'):
 for r in json.loads(p.read_text())['records']:
  if r['qualified']:baseline[(p.parent.name,r['id'])]=r
profiles=[]
for response in ['native','spin_kernel','response_kernel']:
 for xc in ['pbe0','bhandhlyp','lc_wpbe']:
  rows=[]
  for (mol,state),b in baseline.items():
   relax=b['se_relaxation_ev'];frozen=b['dft_frozen_ev']
   if response!='native':
    path=ROOT/response/mol/'result.json'
    if not path.exists():continue
    a=next(r for r in json.loads(path.read_text())['records'] if r['id']==state)
    if not a['qualified']:continue
    relax=a['relaxation_ev']
   if xc!='pbe0':
    path=ROOT/xc/mol/'result.json'
    if not path.exists():continue
    a=next(r for r in json.loads(path.read_text())['records'] if r['id']==state)
    if not a['qualified']:continue
    frozen=a['frozen_ev']
   energy=frozen+relax;rows.append(dict(molecule=mol,state=state,energy_ev=energy,tbe_ev=b['tbe_ev'],
    error_ev=energy-b['tbe_ev'],baseline_ev=b['hybrid_ev'],baseline_error_ev=b['hybrid_error_ev']))
  if rows:profiles.append(dict(response=response,reference=xc,count=len(rows),
   mae_ev=sum(abs(r['error_ev']) for r in rows)/len(rows),
   maximum_absolute_error_ev=max(abs(r['error_ev']) for r in rows),rows=rows))
assert len(baseline)==10
eligible=[p for p in profiles if p['count']==10]
selected=min(eligible,key=lambda p:p['mae_ev'])
write(ROOT/'development_summary.json',dict(host=HOST,profiles=profiles,selected=selected,
 selection='lowest MAE among nine fixed development profiles, ten identical states',
 no_parameters_fitted=True,not_independent_validation=True))
freeze=dict(host=HOST,frozen_utc=datetime.now(timezone.utc).isoformat(),
 selected_reference=selected['reference'],selected_response=selected['response'],
 lambda_exchange=.9,parameters_fitted=0,selection_data='historical ten development states only',
 profile_changes_after_holdout_allowed=False,
 cases=[dict(name='ketene',folder='ketene_1_pbe0',metadata='Ketene.json',metadata_index=0,symmetry='A2',family='ppi'),
  dict(name='diazomethane',folder='diazomethane_1_pbe0',metadata='Diazomethane.json',metadata_index=0,symmetry='A2',family='ppi'),
  dict(name='pyrrole',folder='pyrrole_pbe0',metadata='Pyrrole.json',metadata_index=4,symmetry='B2',family='ppi'),
  dict(name='nitroaniline',folder='nitroaniline_pbe0/full_ao_retry_2',metadata='Nitroaniline.json',metadata_index=0,symmetry='A2',family='npi')],
 state_selection='lowest valence TDA reference state of the specified irrep; compact-GFN2 transition projection norm >= 0.7',
 minimum_csf_overlap=.7,minimum_transition_overlap=.7,minimum_orbital_mapping_overlap=.7,minimum_orbital_mapping_margin=.05,
 saddle_indices=[0,1,2,3],maximum_iterations_per_attempt=100,
 accuracy_gate=dict(minimum_states=3,require_npi=True,require_pipi=True,
  paired_mae_strictly_lower=True,maximum_absolute_error_not_increased=True,maximum_state_regression_ev=.25),
 scripts={p.name:digest(p) for p in BASE.glob('*.py')})
path=ROOT/'holdout_protocol.json'
if path.exists():
 old=json.loads(path.read_text())
 assert (old['selected_reference'],old['selected_response'])==(freeze['selected_reference'],freeze['selected_response'])
else:write(path,freeze)
print('SELECTED',selected['response'],selected['reference'],'MAE',selected['mae_ev'],'MAX',selected['maximum_absolute_error_ev'])
print('HOLDOUT FROZEN',digest(path),flush=True)
