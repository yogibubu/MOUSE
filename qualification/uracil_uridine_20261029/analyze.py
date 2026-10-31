"""Reproduce state assignments and shifts from saved remote wavefunctions."""
from remote_bootstrap import HOST
from benchmark import BASE,write,characterize,EV
from dataclasses import replace
from pathlib import Path
import json,numpy as np
from matrix_apoc import ExcitedState,follow_excited_state,singlet_csf_overlap
from matrix_gfn2.deltascf_derivatives import load_derivative_point
from matrix_link.qmqm_component import CappedQMComponent

def load(name,filename='ground.npz',qualified=False):
 out=BASE/'results'/name;d=json.loads((out/'summary.json').read_text());x=np.array(d['input']['coordinates_angstrom']);path=(out/'qualified' if qualified else out)/filename
 return load_derivative_point(path,request_hash=name,coordinates_angstrom=x),d

def tda_states(name):
 g,d=load(name);s=g.reference_state
 with np.load(BASE/'results'/name/'tda.npz') as f:
  densities=[np.einsum('pi,i,qi->pq',s.coefficients[:,f['occupied']],u,s.coefficients[:,f['virtual']]) for u in f['amplitudes']]
  states=[ExcitedState(k+1,t,energy_hartree=float(e)) for k,(t,e) in enumerate(zip(densities,f['energies']))]
 return s,states,d

records={}
for mol in ['uracil','uridine','capped']:
 name=f'{mol}_s0'
 if not (BASE/'results'/name/'summary.json').exists():continue
 s0,td0,d0=tda_states(name);p,n,_,_=characterize(s0,d0['input']['base_atoms']);occupied=np.flatnonzero(s0.alpha_occupations>.5);virtual=np.flatnonzero(s0.alpha_occupations<.5)
 with np.load(BASE/'results'/name/'tda.npz') as f:
  initial={}
  for target in ['n_pi','pi_pi']:
   hole=max(i for i in occupied if (p[i]>.4 if target=='pi_pi' else n[i]>.3 and p[i]<.3));pair=(f['occupied']==hole)&(f['virtual']==virtual[0]);weights=np.sum(f['amplitudes'][:,pair]**2,axis=1);initial[target]=int(np.argmax(weights))
 for geom in ['s0','n_pi','pi_pi']:
  name=f'{mol}_{geom}'
  if not (BASE/'results'/name/'summary.json').exists():continue
  s,candidates,d=tda_states(name)
  for target in ['n_pi','pi_pi']:
   item=dict(name=name,target=target)
   reference=td0[initial[target]]
   if geom=='s0':selected=reference;item.update(tda_overlap=1.,tda_margin=None)
   else:
    cross=CappedQMComponent._cross_overlap(s0,s)
    selected,match=follow_excited_state(reference,candidates,cross,overlap_reference=s0.overlap,overlap_candidate=s.overlap,minimum_overlap=0.,ambiguity_margin=0.,strict=False)
    item.update(tda_overlap=match.overlap,tda_margin=match.margin)
   item.update(tda_gap_ev=selected.energy_hartree*EV,tda_root=selected.state_id)
   path=BASE/'results'/name/'qualified'/f'roks_{target}.npz'
   if path.exists():
    r,_=load(name,f'roks_{target}.npz',qualified=True);pp,nn,low,_=characterize(s,d['input']['base_atoms'],r.common_coefficients)
    hi=int(np.flatnonzero(r.occupations[1]-r.occupations[0]==1)[0]);pi=int(np.flatnonzero(r.occupations[0]-r.occupations[1]==1)[0]);sugar=[a for a in range(len(d['input']['atomic_numbers'])) if a not in d['input']['base_atoms']];pop=np.sum(low[np.isin(s.ao_atom,sugar)]**2,axis=0)
    item.update(roks_gap_ev=r.excitation_energy_hartree*EV,roks_residual=r.orbital_residual,hole_pi=pp[hi],particle_pi=pp[pi],hole_n=nn[hi],sugar_hole=pop[hi],sugar_particle=pop[pi])
    if geom=='s0':
     seed=replace(r,common_coefficients=s.coefficients);overlap=singlet_csf_overlap(seed.to_apoc(),r.to_apoc(),s.overlap)[0]
    else:
     origin=BASE/'results'/f'{mol}_s0'/'qualified'/f'roks_{target}.npz'
     if origin.exists():
      r0,_=load(f'{mol}_s0',f'roks_{target}.npz',qualified=True);overlap=singlet_csf_overlap(r0.to_apoc(),r.to_apoc(),CappedQMComponent._cross_overlap(s0,s))[0]
     else:overlap=None
    item['roks_overlap']=overlap
    char_ok=pp[pi]>.5 and (pp[hi]>.5 if target=='pi_pi' else pp[hi]<.3 and nn[hi]>.3)
    item['roks_character_qualified']=bool(char_ok)
    item['roks_tracking_qualified']=overlap is not None and overlap>=.7
   else:item['roks_status']='not converged or no accepted endpoint'
   records[f'{name}_{target}']=item
shifts=[]
for geom,target in [('s0','n_pi'),('s0','pi_pi'),('n_pi','n_pi'),('pi_pi','pi_pi')]:
 a=records.get(f'uracil_{geom}_{target}',{});b=records.get(f'uridine_{geom}_{target}',{});cap=records.get(f'capped_{geom}_{target}',{})
 row=dict(geometry=geom,target=target)
 for model in ['tda','roks']:
  key=model+'_gap_ev'
  if key in a and key in b:row[model+'_uridine_minus_uracil_ev']=b[key]-a[key]
  if key in cap and key in b:row[model+'_full_minus_capped_ev']=b[key]-cap[key]
 shifts.append(row)
write(BASE/'comparison.json',dict(host=HOST,states=records,shifts=shifts))
print(json.dumps(shifts,indent=2))
