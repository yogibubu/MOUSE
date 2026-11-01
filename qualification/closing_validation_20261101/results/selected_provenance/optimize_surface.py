"""Optimize the final anchored OO-CID energy, then its analytic Hessian.

All accepted geometry steps and orbital/CI checkpoints are restartable.
The final frequencies are unscaled harmonic values with rigid motions removed.
"""
from pathlib import Path
import sys,socket,runpy,json,argparse,time,hashlib
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
BASE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101');HERE=BASE/'oracle/MOUSE-closing-validation-20261101';REPAIR=BASE/'oracle/MOUSE-pi-pi-transfer-repair-20261101'
runpy.run_path(str(BASE/'source/qualification/remote_bootstrap.py'))
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import null_space
from threadpoolctl import threadpool_limits
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from matrix_gfn2.nosi_surface import PreparedNOSIModel,save_nosi_result,load_nosi_result

def write(path,v):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(v,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else x.item())+'\n');temp.replace(path)

def run(cohort,name):
 out=HERE/'structures'/name;out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():return
 model=PreparedNOSIModel(REPAIR/'prepared_models'/cohort/name/'model.json');calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in model.anchor.atomic_numbers],temperature_kelvin=0.)
 calc.evaluate=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Unexpected SCC during prepared optimization'))
 reference=None;history=json.loads((out/'history.json').read_text()) if (out/'history.json').exists() else []
 if (out/'accepted.npz').exists():reference=load_nosi_result(out/'accepted.npz',model);x=reference.coordinates_angstrom*ANGSTROM_TO_BOHR
 else:
  x=model.anchor.coordinates_bohr.copy();normal=model.manifest['provenance']['reflection_certificate'].get('normal');normal=np.linalg.svd(x-x.mean(0),full_matrices=True)[2][-1] if normal is None else normal;carbon=next(i for i,z in enumerate(model.anchor.atomic_numbers) if z==6);x[carbon]+=np.array(normal)*.15
 cache={};started=time.monotonic()
 def fun(flat):
  if 'x' not in cache or not np.array_equal(flat,cache['x']):
   r=model.evaluate(calc,flat.reshape(x.shape)/ANGSTROM_TO_BOHR,properties=('energy','gradient'),reference_result=reference);s=r.states[0];cache.clear();cache.update(x=flat.copy(),result=r)
   write(out/'trial.json',dict(energy=s.energy_hartree,gradient_max=float(abs(s.gradient_hartree_per_bohr).max()),overlap=s.overlap,coordinates_bohr=flat.reshape(x.shape)))
  s=cache['result'].states[0];return s.energy_hartree,s.gradient_hartree_per_bohr.ravel()
 def callback(flat):
  nonlocal reference
  fun(flat);reference=cache['result'];save_nosi_result(out/'accepted.npz',reference);s=reference.states[0]
  history.append(dict(step=len(history)+1,energy=s.energy_hartree,gradient_max=float(abs(s.gradient_hartree_per_bohr).max()),overlap=s.overlap,coordinates_bohr=flat.reshape(x.shape)));write(out/'history.json',history);print(name,history[-1]['step'],s.energy_hartree,history[-1]['gradient_max'],flush=True)
 result=minimize(fun,x.ravel(),jac=True,method='BFGS',callback=callback,options=dict(gtol=5e-6,maxiter=100))
 fun(result.x);reference=cache['result'];save_nosi_result(out/'optimized.npz',reference)
 gmax=float(abs(reference.states[0].gradient_hartree_per_bohr).max());assert gmax<2e-5,(name,result.message,gmax)
 start=time.monotonic();final=model.evaluate(calc,reference.coordinates_angstrom,properties=('energy','gradient','hessian'),reference_result=reference);hseconds=time.monotonic()-start;save_nosi_result(out/'final_hessian.npz',final)
 s=final.states[0];xyz=final.coordinates_angstrom*ANGSTROM_TO_BOHR;masses=np.array([{1:1.00782503223,6:12.,7:14.00307400443,8:15.99491461957}[int(z)] for z in model.anchor.atomic_numbers]);center=np.average(xyz,axis=0,weights=masses);relative=xyz-center;root=np.sqrt(masses);rigid=[]
 for axis in np.eye(3):rigid.append((root[:,None]*axis).ravel())
 for axis in np.eye(3):rigid.append((root[:,None]*np.cross(axis,relative)).ravel())
 q=null_space(np.array(rigid));mass=np.repeat(masses,3);hw=s.hessian_hartree_per_bohr2/np.sqrt(mass[:,None]*mass[None,:]);values,vectors=np.linalg.eigh(q.T@hw@q);frequencies=np.sign(values)*np.sqrt(abs(values))*5140.487143715828;cart=(q@vectors)/np.sqrt(mass[:,None])
 np.savez_compressed(out/'normal_modes.npz',frequencies_cm=frequencies,cartesian_modes=cart,coordinates_bohr=xyz,masses_amu=masses,hessian=s.hessian_hartree_per_bohr2)
 report=dict(molecule=name,model_fingerprint=model.fingerprint,host=socket.getfqdn(),gradient_max=gmax,frequencies_cm=frequencies,imaginary_modes=int(sum(frequencies < -20)),minimum_frequency_cm=float(frequencies.min()),coordinates_angstrom=final.coordinates_angstrom,atomic_numbers=model.anchor.atomic_numbers,optimization_steps=len(history),optimization_message=str(result.message),wall_seconds=time.monotonic()-started,hessian_seconds=hseconds,maximum_stationary_residual=final.maximum_stationary_residual,minimum_overlap=min(s.overlap for s in final.states),derivatives='analytic of final frozen anchored model; no re-preparation inside optimization',frequency_scaling=1.)
 write(out/'result.json',report);print(json.dumps(report,default=lambda x:x.tolist()),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('cohort');p.add_argument('name');a=p.parse_args()
 with threadpool_limits(2):run(a.cohort,a.name)
