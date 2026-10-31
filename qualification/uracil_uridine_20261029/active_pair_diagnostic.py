from remote_bootstrap import HOST
from benchmark import BASE,write
from dataclasses import replace
import numpy as np,json
from matrix_gfn2.deltascf_derivatives import load_derivative_point
from matrix_link.qmqm_component import CappedQMComponent
records=[]
for mol in ['uracil','uridine','capped']:
 for geom,target in [('s0','n_pi'),('s0','pi_pi'),('n_pi','n_pi'),('pi_pi','pi_pi')]:
  name=f'{mol}_{geom}';out=BASE/'results'/name;path=out/'qualified'/f'roks_{target}.npz'
  if not path.exists():continue
  d=json.loads((out/'summary.json').read_text());x=np.array(d['input']['coordinates_angstrom']);r=load_derivative_point(path,request_hash=name,coordinates_angstrom=x)
  ids=np.flatnonzero(r.occupations[0]!=r.occupations[1]);cr=r.common_coefficients[:,ids]
  if geom=='s0':ref=r.reference_state.coefficients[:,ids];cross=r.overlap
  else:
   origin=BASE/'results'/f'{mol}_s0';p=origin/'qualified'/f'roks_{target}.npz'
   if not p.exists():continue
   od=json.loads((origin/'summary.json').read_text());r0=load_derivative_point(p,request_hash=f'{mol}_s0',coordinates_angstrom=np.array(od['input']['coordinates_angstrom']));oi=np.flatnonzero(r0.occupations[0]!=r0.occupations[1]);ref=r0.common_coefficients[:,oi];cross=CappedQMComponent._cross_overlap(r0.reference_state,r.reference_state)
  block=ref.T@cross@cr;score=abs(block[0,0]*block[1,1]+block[0,1]*block[1,0]);records.append(dict(name=name,target=target,active_pair_overlap=score,cross=block))
write(BASE/'active_pair_diagnostic.json',records)
print(json.dumps(records,default=lambda a:a.tolist(),indent=2))
