"""Lightweight sensitivity of fixed transition weights and vertical parent mix.
Not a recomputation of DFT at displaced anchor geometries.
"""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];Q=ROOT/'qualification/pi_pi_repair_20261031'
data=json.loads((Q/'effective_rank_development.json').read_text());rank_rows=[]
for r in data['rows']:
 w=np.array(r['weights']);a=w[0];b=w[1:].sum();c=w[1:]@w[1:];crossings=[]
 for boundary in [r['rank']-.5,r['rank']+.5]:
  if boundary<1:continue
  for u in np.roots([b*b-boundary*c,2*a*b,a*a*(1-boundary)]):
   if abs(u.imag)<1e-8 and u.real>0:
    scale=float(np.sqrt(u.real));crossings.append(dict(boundary=boundary,subleading_singular_value_scale=scale,relative_change_percent=100*abs(scale-1)))
 nearest=min(crossings,key=lambda x:x['relative_change_percent']) if crossings else None
 rank_rows.append(dict(state=r['state'],molecule=r['molecule'],rank=r['rank'],effective_pairs=r['participation'],nearest_crossing=nearest))
choices={'B016':('Cyclopentadiene','0','1'),'B022':('Furan','0','1'),'B024':('Phthalimide','0_fr_v1','1_gmf_polish_v1')};mix=[]
for sid,(name,*attempts) in choices.items():
 parents=[json.loads((Q/f'remote_snapshot/pi_pi_repair/optimized/{name}/{sid}/{p}/result.json').read_text())['excitation_ev'] for p in attempts];original=next(r for r in data['rows'] if r['state']==sid)
 mix.append(dict(state=sid,molecule=name,parent_energies_ev=parents,vertical_energies_ev={str(w):w*parents[0]+(1-w)*parents[1] for w in [.4,.5,.6]},rank_one_ev=original['baseline_ev'],rank_two_ev=original['candidate_ev'],vertical_rank_switch_jump_ev=original['candidate_ev']-original['baseline_ev']))
out=dict(scope='exposed development; scalar perturbation of fixed reference NTO singular values and algebraic vertical weight sensitivity, no new anchor or nuclear surface',rank_rule='nearest integer inverse NTO-weight purity',production_weights=[.5,.5],rank_sensitivity=rank_rows,weight_sensitivity=mix)
p=Path(__file__).with_name('rank_weight_sensitivity.json');p.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(next(r for r in rank_rows if r['state']=='B024'),indent=2));print(json.dumps(mix,indent=2))
