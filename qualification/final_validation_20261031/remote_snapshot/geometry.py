"""Optimize a frozen multi-CSF PES and characterize its analytic Hessian."""
from common import *
from scipy.optimize import minimize
from scipy.linalg import null_space
import fcntl,argparse,resource

def vibrations(point):
 s=point.reference_state;xyz=s.coordinates_bohr
 masses=np.array([{1:1.00782503223,6:12.,7:14.00307400443,8:15.99491461957}[int(z)] for z in s.atomic_numbers]);root=np.repeat(np.sqrt(masses),3)
 center=np.average(xyz,axis=0,weights=masses);r=xyz-center;rigid=[]
 for axis in np.eye(3):rigid.extend([(np.ones_like(r)*axis*np.sqrt(masses)[:,None]).ravel(),(np.cross(axis,r)*np.sqrt(masses)[:,None]).ravel()])
 u,sv,_=np.linalg.svd(np.array(rigid).T,full_matrices=True);rank=int(sum(sv>1e-8));q=u[:,rank:];hh=point.hessian_hartree_per_bohr2/root[:,None]/root[None,:]
 values,vectors=eigh(q.T@hh@q);modes=q@vectors
 freq=np.sign(values)*np.sqrt(abs(values))*5140.487143715828
 return freq,modes,masses

def run(name,sid):
 model=MultiCSFModel(W/f'models/{name}/{sid}/model.json');out=OUT/f'geometry/{name}';out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():return
 calc=calculator(model.anchor);calc.evaluate=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Unexpected SCC in fixed surface'))
 accepted=out/'accepted.json';history=json.loads((out/'history.json').read_text()) if (out/'history.json').exists() else []
 reference=load_multicsf_result(out/json.loads(accepted.read_text())['path']/'result.json',model,calc) if accepted.exists() else None
 x=model.xyz if reference is None else reference.reference_state.coordinates_bohr/ANGSTROM_TO_BOHR
 cache={};start=time.monotonic()
 def objective(flat):
  nonlocal reference
  if 'x' not in cache or not np.array_equal(cache['x'],flat):
   point=evaluate_multicsf_surface(calc,flat.reshape(-1,3)/ANGSTROM_TO_BOHR,model,reference_result=reference,compute_hessian=False)
   cache.update(x=flat.copy(),point=point)
  point=cache['point'];return point.energy_hartree,point.gradient_hartree_per_bohr.ravel()
 def callback(flat):
  nonlocal reference
  objective(flat);point=cache['point'];index=len(history);dest=out/f'accepted_{index:03d}'
  save_multicsf_result(dest,point,model);write(accepted,dict(path=dest.name));reference=point
  history.append(dict(step=index,energy=point.energy_hartree,max_gradient=float(np.max(abs(point.gradient_hartree_per_bohr))),fidelity=point.fidelity,seconds=time.monotonic()-start));write(out/'history.json',history)
 result=minimize(objective,x.ravel()*ANGSTROM_TO_BOHR,jac=True,method='BFGS',callback=callback,options=dict(gtol=3e-5,maxiter=80))
 objective(result.x);point=cache['point'];gmax=float(np.max(abs(point.gradient_hartree_per_bohr)))
 if gmax>3e-5:raise RuntimeError('Geometry not stationary: '+str(result.message)+' max gradient '+str(gmax))
 cp=out/'analytic_stationary'
 if (cp/'result.json').exists():final=load_multicsf_result(cp/'result.json',model,calc)
 else:final=evaluate_multicsf_surface(calc,result.x.reshape(-1,3)/ANGSTROM_TO_BOHR,model,reference_result=point,compute_hessian=True);save_multicsf_result(cp,final,model)
 freq,modes,masses=vibrations(final);np.savez_compressed(out/'vibrations.npz',frequencies_cm1=freq,mass_weighted_modes=modes,masses_amu=masses,coordinates_angstrom=result.x.reshape(-1,3)/ANGSTROM_TO_BOHR)
 row=dict(status='stationary_characterized',host=HOST,molecule=name,state=sid,max_gradient=gmax,frequencies_cm1=freq,imaginary_below_minus20=int(sum(freq< -20)),nuclear_minimum=bool(np.all(freq>=-20)),electronic_orbital_index=final.orbital_saddle_index,model_fingerprint=model.fingerprint,frozen_rank=model.rank,analytic_hessian=True,seconds_current_run=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,physical_reference_comparison='separate required step; an imaginary nuclear mode is retained, never clipped')
 write(out/'result.json',row);print(name,'STATIONARY',gmax,'frequencies',freq,flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('name');a.add_argument('sid');args=a.parse_args();lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 with threadpool_limits():run(args.name,args.sid)
