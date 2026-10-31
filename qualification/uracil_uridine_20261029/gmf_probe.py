from remote_bootstrap import HOST
from benchmark import BASE,write,characterize,EV
from types import SimpleNamespace
from dataclasses import replace
import json,time,argparse,numpy as np
from matrix_gfn2 import GFN2Calculator,evaluate_gfn2_roks_singlet
from matrix_gfn2.roks import ROKSSingletResult
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.deltascf_derivatives import load_derivative_point,save_derivative_point

def run(name,target,index,seed,freeze):
 out=BASE/'results'/name;meta=json.loads((out/'summary.json').read_text());xyz=np.array(meta['input']['coordinates_angstrom'])
 calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[z] for z in meta['input']['atomic_numbers']],temperature_kelvin=0.)
 gs=load_derivative_point(out/'ground.npz',request_hash=name,coordinates_angstrom=xyz);s=gs.reference_state;ground=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=s.total_energy_hartree)
 ctx=prepare_gfn2_deltascf(calc,xyz,ground_state=ground)
 p,n,_,_=characterize(s,meta['input']['base_atoms']);occupied=np.flatnonzero(s.alpha_occupations>.5);holes=[i for i in occupied if (p[i]>.4 if target=='pi_pi' else n[i]>.3 and p[i]<.3)]
 hole=max(holes);particle=int(np.flatnonzero(s.alpha_occupations<.5)[0]);occ=np.array([s.alpha_occupations,s.beta_occupations]);occ[0,hole]=0;occ[0,particle]=1;c=s.coefficients.copy()
 folder=out/f'gmf_{target}_{index}_{seed}';folder.mkdir(exist_ok=True)
 destination=folder/'converged.npz';key=f'{name}_gmf_{target}_{index}_{seed}'
 if destination.exists():return
 if (folder/'latest.npz').exists():
  with np.load(folder/'latest.npz') as f:c=f['coefficients'];occ=f['occupations']
 elif seed=='trial' and (out/'qualified'/f'trial_{target}.npz').exists():
  with np.load(out/'qualified'/f'trial_{target}.npz') as f:c=f['common_coefficients'];occ=f['occupations']
 triplet=np.array([np.maximum(*occ),np.minimum(*occ)]);pm=np.array([(c*o)@c.T for o in occ]);pt=np.array([(c*o)@c.T for o in triplet]);fm,em=ctx.fock_and_energy(pm);ft,et=ctx.fock_and_energy(pt)
 ref=ROKSSingletResult(c,occ,triplet,s.overlap,fm,ft,2*em-et,float('1'),0,False,reference_state=s,core_electrons=np.asarray(s.atomic_numbers)-np.asarray(calc._valence,dtype=int),ground_energy_hartree=gs.energy_hartree)
 def checkpoint(result):
  np.savez_compressed(folder/'latest.npz',coefficients=result.point,occupations=occ,coordinates_angstrom=xyz)
  write(folder/'history.json',list(result.history))
 t=time.perf_counter()
 try:
  r=evaluate_gfn2_roks_singlet(calc,xyz,*occ,ground_state=ground,reference_result=ref,minimum_state_overlap=0.,gradient=True,orbital_solver='gmf',saddle_index=index,maximum_iterations=60,maximum_step=.2,orbital_tolerance=1e-9,freeze_and_release=freeze,freeze_iterations=8,checkpoint=checkpoint)
  save_derivative_point(destination,r,request_hash=key,coordinates_angstrom=xyz)
  pp,nn,_,_=characterize(s,meta['input']['base_atoms'],r.common_coefficients);hi=int(np.flatnonzero(r.occupations[1]-r.occupations[0]==1)[0]);pi=int(np.flatnonzero(r.occupations[0]-r.occupations[1]==1)[0])
  result=dict(host=HOST,name=name,target=target,seconds=time.perf_counter()-t,gap_ev=r.excitation_energy_hartree*EV,index=r.orbital_saddle_index,residual=r.orbital_residual,hole_pi=pp[hi],particle_pi=pp[pi],hole_n=nn[hi],request_hash=key)
 except Exception as error:result=dict(host=HOST,name=name,target=target,seconds=time.perf_counter()-t,error=str(error),checkpoint=str(folder/'latest.npz'))
 write(folder/'summary.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('name');a.add_argument('target');a.add_argument('index',type=int);a.add_argument('--seed',default='trial');a.add_argument('--freeze',action='store_true');args=a.parse_args();run(args.name,args.target,args.index,args.seed,args.freeze)
