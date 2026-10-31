from remote_bootstrap import HOST
from benchmark import BASE,write,characterize,EV
from pathlib import Path
from dataclasses import replace,fields
from types import SimpleNamespace
import argparse,json,time,traceback,numpy as np
from matrix_gfn2 import GFN2Calculator,evaluate_gfn2_roks_singlet
from matrix_gfn2.roks import ROKSSingletResult,ROKSConvergenceError
from matrix_gfn2.deltascf_derivatives import load_derivative_point,save_derivative_point

def run(name,target,solver="least_squares"):
 out=BASE/'results'/name;d=json.loads((out/'summary.json').read_text());xyz=np.array(d['input']['coordinates_angstrom']);qualified=out/'qualified';qualified.mkdir(exist_ok=True)
 destination=qualified/f'roks_{target}.npz'
 if destination.exists():return
 calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[z] for z in d['input']['atomic_numbers']],temperature_kelvin=0.)
 gs=load_derivative_point(out/'ground.npz',request_hash=name,coordinates_angstrom=xyz);s=gs.reference_state
 ground=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=s.total_energy_hartree)
 ref=None
 p,n,_,_=characterize(s,d['input']['base_atoms'])
 occupied=np.flatnonzero(s.alpha_occupations>.5);virtual=np.flatnonzero(s.alpha_occupations<.5)
 holes=[i for i in occupied if (p[i]>.4 if target=='pi_pi' else n[i]>.3 and p[i]<.3)]
 hole=max(holes);particle=int(virtual[0]);alpha=s.alpha_occupations.copy();beta=s.beta_occupations.copy();alpha[hole]=0;alpha[particle]=1
 # Follow accepted S0 CSF at the same molecule's other two geometries.
 if not name.endswith('_s0'):
  start=BASE/'results'/f"{name.split('_')[0]}_s0"/'qualified'/f'roks_{target}.npz'
  if start.exists():
   origin=json.loads((start.parent.parent/'summary.json').read_text());ref=load_derivative_point(start,request_hash=start.parent.parent.name,coordinates_angstrom=np.array(origin['input']['coordinates_angstrom']));alpha,beta=ref.occupations
 trialfile=qualified/f'trial_{target}.npz'
 if trialfile.exists():
  with np.load(trialfile,allow_pickle=False) as f:
   kw={key:f[key].item() if f[key].ndim==0 else f[key] for key in f.files}
  ref=ROKSSingletResult(**kw,reference_state=s,core_electrons=np.asarray(s.atomic_numbers)-np.asarray(calc._valence,dtype=int),ground_energy_hartree=gs.energy_hartree);alpha,beta=ref.occupations
 historyfile=qualified/f'{target}.json';history=json.loads(historyfile.read_text()) if historyfile.exists() else dict(host=HOST,name=name,target=target,attempts=[])
 t=time.perf_counter()
 try:
  r=evaluate_gfn2_roks_singlet(calc,xyz,alpha,beta,ground_state=ground,reference_result=ref,gradient=True,minimum_state_overlap=0.,orbital_tolerance=1e-9,maximum_evaluations=120,orbital_solver=solver,**({'rechart_interval':6} if solver=='least_squares' else {}))
  save_derivative_point(destination,r,request_hash=name,coordinates_angstrom=xyz)
  pp,nn,low,_=characterize(s,d['input']['base_atoms'],r.common_coefficients)
  hole=int(np.flatnonzero(r.occupations[1]-r.occupations[0]==1)[0]);part=int(np.flatnonzero(r.occupations[0]-r.occupations[1]==1)[0])
  history['attempts'].append(dict(solver=solver,seconds=time.perf_counter()-t,energy=r.energy_hartree,gap_ev=r.excitation_energy_hartree*EV,residual=r.orbital_residual,hole_pi=pp[hole],particle_pi=pp[part],hole_n=nn[hole],evaluations=r.evaluations))
 except ROKSConvergenceError as error:
  r=error.result;np.savez_compressed(trialfile,**{f.name:getattr(r,f.name) for f in fields(r) if getattr(r,f.name) is not None and f.name not in ['reference_state','core_electrons','ground_energy_hartree']})
  history['attempts'].append(dict(seconds=time.perf_counter()-t,error=str(error),residual=r.orbital_residual,checkpoint=str(trialfile)))
 except Exception as error:history['attempts'].append(dict(seconds=time.perf_counter()-t,error=str(error),traceback=traceback.format_exc()))
 write(historyfile,history)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('name');a.add_argument('target');a.add_argument('--solver',default='least_squares');args=a.parse_args();run(args.name,args.target,args.solver)
