from remote_bootstrap import HOST
from benchmark import BASE,write,characterize,EV
from pathlib import Path
from dataclasses import replace
import numpy as np,json,time,traceback,argparse
from matrix_gfn2 import GFN2Calculator,evaluate_gfn2_deltascf_analytic_hessian
from matrix_gfn2.stda import compact_xtb_tda
from matrix_gfn2.deltascf_derivatives import load_derivative_point

def run(name,target,qualified=False):
 out=BASE/'results'/name;d=json.loads((out/'summary.json').read_text());xyz=np.array(d['input']['coordinates_angstrom'])
 calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[z] for z in d['input']['atomic_numbers']],temperature_kelvin=0.)
 stateout=out/'qualified' if qualified else out
 path=stateout/f'roks_{target}.npz'
 if not path.exists():return
 r=load_derivative_point(path,request_hash=name,coordinates_angstrom=xyz)
 report=stateout/f'derivatives_{target}.json'
 record=json.loads(report.read_text()) if report.exists() else dict(host=HOST,name=name,target=target)
 seed=r.reference_state
 p,n,low,normal=characterize(seed,d['input']['base_atoms'],r.common_coefficients)
 hole=int(np.flatnonzero(r.occupations[1]-r.occupations[0]==1)[0]);particle=int(np.flatnonzero(r.occupations[0]-r.occupations[1]==1)[0])
 sugar=[a for a in range(len(xyz)) if a not in d['input']['base_atoms']]
 pop=np.sum(low[np.isin(seed.ao_atom,sugar)]**2,axis=0)
 record['character']=dict(hole_pi=p[hole],particle_pi=p[particle],hole_oxygen_inplane=n[hole],sugar_hole=pop[hole],sugar_electron=pop[particle])
 write(report,record)
 if 'roks_hessian_seconds' not in record:
  t=time.perf_counter();h=evaluate_gfn2_deltascf_analytic_hessian(calc,xyz,*r.occupations,mode='roks',reference_result=r)
  np.savez_compressed(stateout/f'roks_hessian_{target}.npz',hessian=h.hessian_hartree_per_bohr2,gradient=r.gradient_hartree_per_bohr,orbital_hessian=h.orbital_hessian_hartree)
  record['roks_hessian_seconds']=time.perf_counter()-t;record['roks_response_residual']=h.maximum_response_residual;record['roks_asymmetry']=h.raw_symmetry_error_hartree_per_bohr2;record['roks_curvatures']=np.linalg.eigvalsh(h.hessian_hartree_per_bohr2)
  write(report,record)
 if 'tda_gradient_seconds' not in record:
  t=time.perf_counter();exc=compact_xtb_tda(seed,roots=10,virtual_count=int(np.count_nonzero(seed.alpha_occupations<.5)),shift_model='constant');op=calc.closed_shell_stda_hessian(seed,excited=exc,root=d['assignments'][target]['root']-1,spectral_range_ev=10.)
  ground=load_derivative_point(out/'ground.npz',request_hash=name,coordinates_angstrom=xyz)
  g=ground.gradient_hartree_per_bohr+op.excitation_gradient_hartree_per_bohr
  record['tda_gradient_seconds']=time.perf_counter()-t;record['gradient_difference_max']=np.max(np.abs(g-r.gradient_hartree_per_bohr));write(report,record)
  t=time.perf_counter();hh=op.materialize(symmetrize=False)
  record['tda_hessian_seconds']=time.perf_counter()-t;record['tda_asymmetry']=np.max(np.abs(hh-hh.T));record['hessian_difference_max']=np.max(np.abs(hh-h.hessian_hartree_per_bohr2)) if 'h' in locals() else None
  np.savez_compressed(stateout/f'tda_hessian_{target}.npz',hessian=hh,gradient=g)
  write(report,record)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('name');a.add_argument('target');a.add_argument('--qualified',action='store_true');args=a.parse_args();run(args.name,args.target,args.qualified)
