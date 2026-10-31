from remote_bootstrap import HOST
from benchmark import BASE,write
from matrix_gfn2 import GFN2Calculator,evaluate_gfn2_roks_singlet,evaluate_gfn2_deltascf_analytic_hessian
from matrix_gfn2.deltascf_derivatives import load_derivative_point,save_derivative_point
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
import numpy as np,json,time
for name in ['uridine_s0','uridine_n_pi']:
 out=BASE/'results'/name;folder=out/'gmf_n_pi_1_ground';meta=json.loads((out/'summary.json').read_text());summary=json.loads((folder/'summary.json').read_text());xyz=np.array(meta['input']['coordinates_angstrom']);key=summary['request_hash']
 r=load_derivative_point(folder/'converged.npz',request_hash=key,coordinates_angstrom=xyz);calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[z] for z in meta['input']['atomic_numbers']],temperature_kelvin=0.)
 report=folder/'derivatives.json';record=json.loads(report.read_text()) if report.exists() else dict(host=HOST,name=name,target='n_pi',orbital_index=r.orbital_saddle_index)
 hp=folder/'hessian.npz'
 if hp.exists():h=np.load(hp)['hessian']
 else:
  t=time.perf_counter();result=evaluate_gfn2_deltascf_analytic_hessian(calc,xyz,*r.occupations,mode='roks',reference_result=r);h=result.hessian_hartree_per_bohr2
  record.update(hessian_seconds=time.perf_counter()-t,response_residual=result.maximum_response_residual,asymmetry=result.raw_symmetry_error_hartree_per_bohr2);np.savez_compressed(hp,hessian=h,gradient=r.gradient_hartree_per_bohr);write(report,record)
 if 'gradient_direction_error' in record:continue
 v=np.arange(1,xyz.size+1,dtype=float).reshape(xyz.shape);v-=v.mean(axis=0);v/=np.linalg.norm(v);step=2e-4;points=[]
 for sign in [-1,1]:
  x=xyz+sign*step*v/ANGSTROM_TO_BOHR;path=folder/f'validation_{sign:+d}.npz'
  if path.exists():point=load_derivative_point(path,request_hash=key,coordinates_angstrom=x)
  else:
   point=evaluate_gfn2_roks_singlet(calc,x,*r.occupations,gradient=True,reference_result=r,orbital_solver='gmf',saddle_index=1,maximum_iterations=12,orbital_tolerance=1e-10)
   save_derivative_point(path,point,request_hash=key,coordinates_angstrom=x)
  points.append(point)
 efd=(points[1].energy_hartree-points[0].energy_hartree)/(2*step);gdir=float(np.sum(v*r.gradient_hartree_per_bohr));hfd=(points[1].gradient_hartree_per_bohr-points[0].gradient_hartree_per_bohr).ravel()/(2*step)
 record.update(gradient_direction_error=abs(efd-gdir),hessian_action_max_error=float(np.max(abs(h@v.ravel()-hfd))),step_bohr=step)
 assert record['gradient_direction_error']<1e-7 and record['hessian_action_max_error']<1e-5,record
 write(report,record);print(json.dumps(record),flush=True)
