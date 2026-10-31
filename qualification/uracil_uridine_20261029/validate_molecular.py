from remote_bootstrap import HOST
from benchmark import BASE,write
import argparse,json,time,numpy as np
from matrix_gfn2 import GFN2Calculator,evaluate_gfn2_roks_singlet
from matrix_gfn2.deltascf_derivatives import load_derivative_point,save_derivative_point
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR

def run(name,target):
 out=BASE/'results'/name;d=json.loads((out/'summary.json').read_text());xyz=np.array(d['input']['coordinates_angstrom'])
 folder=out/'qualified' if name.startswith('uridine') else out
 path=folder/f'roks_{target}.npz'
 if not path.exists():return
 r=load_derivative_point(path,request_hash=name,coordinates_angstrom=xyz)
 hp=folder/f'roks_hessian_{target}.npz'
 if not hp.exists():return
 report=folder/f'validation_{target}.json'
 if report.exists():return
 calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[z] for z in d['input']['atomic_numbers']],temperature_kelvin=0.)
 v=np.arange(1,xyz.size+1,dtype=float).reshape(xyz.shape);v-=v.mean(axis=0);v/=np.linalg.norm(v)
 step=2e-4;points=[];elapsed=[]
 for sign in [-1,1]:
  x=xyz+sign*step*v/ANGSTROM_TO_BOHR;p=folder/f'validation_{target}_{sign:+d}.npz'
  t=time.perf_counter()
  if p.exists():rr=load_derivative_point(p,request_hash=name+target,coordinates_angstrom=x)
  else:
   rr=evaluate_gfn2_roks_singlet(calc,x,*r.occupations,reference_result=r,gradient=True,orbital_tolerance=1e-10,maximum_evaluations=80,rechart_interval=6)
   save_derivative_point(p,rr,request_hash=name+target,coordinates_angstrom=x)
  points.append(rr);elapsed.append(time.perf_counter()-t)
 fd=(points[1].energy_hartree-points[0].energy_hartree)/(2*step)
 directional=float(np.sum(v*r.gradient_hartree_per_bohr))
 hvp=(points[1].gradient_hartree_per_bohr-points[0].gradient_hartree_per_bohr).ravel()/(2*step)
 analytic=np.load(hp)['hessian']@v.ravel()
 record=dict(host=HOST,step_bohr=step,gradient_direction_error=abs(fd-directional),hessian_action_max_error=float(np.max(abs(analytic-hvp))),point_seconds=elapsed,point_residuals=[p.orbital_residual for p in points])
 assert record['gradient_direction_error']<1e-7 and record['hessian_action_max_error']<1e-5,record
 write(report,record)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('name');a.add_argument('target');args=a.parse_args();run(args.name,args.target)
