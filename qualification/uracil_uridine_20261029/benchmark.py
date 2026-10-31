"""Remote-only, checkpointed DeltaSCF/TDA molecular qualification."""
from remote_bootstrap import HOST
from pathlib import Path
import argparse,json,time,traceback,os
import numpy as np
from matrix_gfn2 import GFN2Calculator,evaluate_gfn2_deltascf,evaluate_gfn2_roks_singlet
from matrix_gfn2.response_reference import stationary_response_reference
from matrix_gfn2.stda import grimme_xtb_stda,compact_xtb_tda
from matrix_gfn2.deltascf_derivatives import save_derivative_point,load_derivative_point
BASE=Path(__file__).resolve().parent
EV=27.211386245988

def write(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else str(a))+'\n');tmp.replace(path)

def characterize(s,base,coefficients=None):
 # Orthogonal AO populations, projected on the fitted heavy-base plane.
 vals,u=np.linalg.eigh(s.overlap);low=(u*np.sqrt(vals))@u.T@(s.coefficients if coefficients is None else coefficients)
 heavy=[i for i in base if s.atomic_numbers[i]>1]
 xyz=s.coordinates_bohr[heavy];_,_,v=np.linalg.svd(xyz-xyz.mean(axis=0));normal=v[-1]
 p=np.zeros(s.nmo);oxygen_inplane=np.zeros(s.nmo)
 # For s/p H,C,N,O basis the spherical transform is orthogonal and square.
 cart=s.basis.spherical_to_cartesian@low
 for atom in heavy:
  ids=[np.flatnonzero((s.basis.cartesian_atom==atom)&(s.basis.cartesian_type==t)) for t in (2,3,4)]
  if not all(len(a)==1 for a in ids):raise ValueError('ambiguous p shell')
  block=cart[[a[0] for a in ids]];perp=(normal@block)**2
  p+=perp
  if s.atomic_numbers[atom]==8:oxygen_inplane+=np.sum(block*block,axis=0)-perp
 return p,oxygen_inplane,low,normal

def run(name,mode):
 d=json.loads((BASE/'inputs'/f'{name}.json').read_text());xyz=np.array(d['coordinates_angstrom']);atoms=d['atomic_numbers']
 out=BASE/'results'/name;out.mkdir(parents=True,exist_ok=True)
 record=json.loads((out/'summary.json').read_text()) if (out/'summary.json').exists() else dict(name=name,host=HOST,threads={k:os.getenv(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']},input=d)
 calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[z] for z in atoms],temperature_kelvin=0.)
 # A converged derivative checkpoint retains the original ground orbital state.
 prior=next(iter(sorted(out.glob('ground.npz'))),None)
 if prior:
  from types import SimpleNamespace
  saved=load_derivative_point(prior,request_hash=name,coordinates_angstrom=xyz)
  ground=SimpleNamespace(converged=True,orbital_state=saved.reference_state,energy_hartree=saved.reference_state.total_energy_hartree)
 else:
  t=time.perf_counter();ground,cert=stationary_response_reference(calc,xyz)
  record['ground_seconds']=time.perf_counter()-t;record['ground_certificate']=cert
 s=ground.orbital_state
 if prior is None:
  gs=evaluate_gfn2_deltascf(calc,xyz,s.alpha_occupations,s.beta_occupations,ground_state=ground,gradient=True,selection='aufbau',density_tolerance=1e-10,orbital_tolerance=1e-10,energy_tolerance=1e-12)
  save_derivative_point(out/'ground.npz',gs,request_hash=name,coordinates_angstrom=xyz)
 t=time.perf_counter();raw=grimme_xtb_stda(s,roots=10,spectral_range_ev=10.)
 record['raw_std2_kernel']=dict(seconds=time.perf_counter()-t,gaps_ev=raw.excitation_energies_hartree*EV)
 t=time.perf_counter();tda=compact_xtb_tda(s,roots=10,virtual_count=int(np.count_nonzero(s.alpha_occupations<.5)),shift_model='constant')
 tda_seconds=time.perf_counter()-t
 p,n,low,normal=characterize(s,d['base_atoms'])
 pop=np.array([np.sum(low[s.ao_atom==a]**2,axis=0) for a in range(len(atoms))])
 sugar=[a for a in range(len(atoms)) if a not in d['base_atoms']]
 roots=[]
 for k,(e,x) in enumerate(zip(tda.excitation_energies_hartree,tda.amplitudes)):
  weights=x*x;oi=tda.occupied_orbitals;vi=tda.virtual_orbitals
  pair=int(np.argmax(weights))
  roots.append(dict(root=k+1,energy_ev=e*EV,hole_pi=float(weights@p[oi]),particle_pi=float(weights@p[vi]),hole_oxygen_inplane=float(weights@n[oi]),dominant_weight=float(weights[pair]),hole=int(oi[pair]),particle=int(vi[pair]),sugar_hole=float(weights@pop[sugar].sum(axis=0)[oi]) if sugar else 0.,sugar_electron=float(weights@pop[sugar].sum(axis=0)[vi]) if sugar else 0.))
 record['tda']=dict(seconds=tda_seconds,roots=roots,primary_count=tda.primary_count,perturbative_count=tda.perturbative_count,solver=tda.solver_backend)
 # Explicit orbital characters, not root numbers, assign initial n/pi seeds.
 eligible=[r for r in roots if r['particle_pi']>.35]
 nroot=max(eligible,key=lambda r:r['hole_oxygen_inplane']*(1-r['hole_pi']))
 proot=max(eligible,key=lambda r:r['hole_pi']*r['particle_pi'])
 record['assignments']={'n_pi':nroot,'pi_pi':proot};record['orbital_diagnostics']=dict(pi=p,oxygen_inplane=n,normal=normal)
 write(out/'summary.json',record)
 np.savez_compressed(out/'tda.npz',energies=tda.excitation_energies_hartree,amplitudes=tda.amplitudes,occupied=tda.occupied_orbitals,virtual=tda.virtual_orbitals)
 if mode=='inspect':return
 for target,root in record['assignments'].items():
  alpha=s.alpha_occupations.copy();beta=s.beta_occupations.copy();alpha[root['hole']]=0;alpha[root['particle']]=1
  checkpoint=out/f'bs_{target}.npz'
  if checkpoint.exists():bs=load_derivative_point(checkpoint,request_hash=name,coordinates_angstrom=xyz)
  else:
   bs=None
   t=time.perf_counter()
   try:
    bs=evaluate_gfn2_deltascf(calc,xyz,alpha,beta,ground_state=ground,gradient=True,maximum_iterations=1000,orbital_tolerance=1e-9,density_tolerance=1e-9,energy_tolerance=1e-12)
    save_derivative_point(checkpoint,bs,request_hash=name,coordinates_angstrom=xyz)
    record[f'bs_{target}']=dict(seconds=time.perf_counter()-t,energy=bs.energy_hartree,gap_ev=bs.excitation_energy_hartree*EV,s2=bs.spin_squared,residual=bs.orbital_residual)
   except Exception as error:record[f'bs_{target}']=dict(error=str(error),seconds=time.perf_counter()-t);write(out/'summary.json',record)
   write(out/'summary.json',record)
  if mode=='bs':continue
  checkpoint=out/f'roks_{target}.npz'
  if checkpoint.exists():continue
  t=time.perf_counter()
  try:
   r=evaluate_gfn2_roks_singlet(calc,xyz,alpha,beta,ground_state=ground,reference_result=bs,gradient=True,orbital_tolerance=1e-9,maximum_evaluations=100)
   save_derivative_point(checkpoint,r,request_hash=name,coordinates_angstrom=xyz)
   record[f'roks_{target}']=dict(seconds=time.perf_counter()-t,energy=r.energy_hartree,gap_ev=r.excitation_energy_hartree*EV,s2=r.spin_squared,residual=r.orbital_residual,evaluations=r.evaluations)
  except Exception as error:record[f'roks_{target}']=dict(error=str(error),traceback=traceback.format_exc(),seconds=time.perf_counter()-t)
  write(out/'summary.json',record)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('name');a.add_argument('--mode',choices=['inspect','bs','roks'],default='bs');args=a.parse_args();run(args.name,args.mode)
