"""Checkpointed symmetry-breaking descent on the unchanged frozen excited PES."""
from common import *
from geometry import vibrations
from scipy.optimize import minimize
import argparse,fcntl,resource,traceback

def run(name,sid):
 model=MultiCSFModel(W/f'models/{name}/{sid}/model.json');calc=calculator(model.anchor)
 calc.evaluate=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Unexpected SCC in fixed surface'))
 source=OUT/f'geometry/{name}';out=OUT/f'geometry_escape/{name}';out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():return
 reference=load_multicsf_result(source/'analytic_stationary/result.json',model,calc)
 initial_energy=reference.energy_hartree;start=time.monotonic()
 for stage in range(3):
  dest=out/f'stage_{stage}';dest.mkdir(exist_ok=True);accepted=dest/'accepted.json'
  history=json.loads((dest/'history.json').read_text()) if (dest/'history.json').exists() else []
  if accepted.exists():reference=load_multicsf_result(dest/json.loads(accepted.read_text())['path']/'result.json',model,calc);x=reference.reference_state.coordinates_bohr.ravel().copy()
  else:
   freq,modes,masses=vibrations(reference);direction=modes[:,0]/np.repeat(np.sqrt(masses),3);direction/=np.linalg.norm(direction)
   x=reference.reference_state.coordinates_bohr.ravel()+.08*direction
   write(dest/'initial_displacement.json',dict(mode_frequency_cm1=float(freq[0]),norm_bohr=.08,reference_energy_hartree=reference.energy_hartree,model_fingerprint=model.fingerprint,source_sha256=sha(__file__)))
  cache={}
  def objective(flat):
   if 'x' not in cache or not np.array_equal(cache['x'],flat):
    point=evaluate_multicsf_surface(calc,flat.reshape(-1,3)/ANGSTROM_TO_BOHR,model,reference_result=reference,compute_hessian=False);cache.update(x=flat.copy(),point=point)
   return cache['point'].energy_hartree,cache['point'].gradient_hartree_per_bohr.ravel()
  def callback(flat):
   nonlocal reference
   objective(flat);point=cache['point'];folder=dest/f'accepted_{len(history):03d}';save_multicsf_result(folder,point,model);write(accepted,dict(path=folder.name));reference=point
   history.append(dict(energy_hartree=point.energy_hartree,max_gradient=float(np.max(abs(point.gradient_hartree_per_bohr))),fidelity=point.fidelity,elapsed_seconds=time.monotonic()-start));write(dest/'history.json',history)
  result=minimize(objective,x,jac=True,method='BFGS',callback=callback,options=dict(gtol=3e-5,maxiter=max(1,60-len(history))))
  objective(result.x);point=cache['point'];gmax=float(np.max(abs(point.gradient_hartree_per_bohr)))
  if gmax>3e-5:raise RuntimeError('Descent budget/identity boundary: '+str(result.message)+'; max gradient '+str(gmax))
  folder=dest/'analytic_stationary'
  if (folder/'result.json').exists():reference=load_multicsf_result(folder/'result.json',model,calc)
  else:reference=evaluate_multicsf_surface(calc,result.x.reshape(-1,3)/ANGSTROM_TO_BOHR,model,reference_result=point,compute_hessian=True);save_multicsf_result(folder,reference,model)
  freq,modes,masses=vibrations(reference);np.savez_compressed(dest/'vibrations.npz',frequencies_cm1=freq,mass_weighted_modes=modes,masses_amu=masses,coordinates_angstrom=result.x.reshape(-1,3)/ANGSTROM_TO_BOHR)
  write(dest/'result.json',dict(max_gradient=gmax,frequencies_cm1=freq,nuclear_minimum=bool(np.all(freq>=-20)),energy_hartree=reference.energy_hartree))
  if np.all(freq>=-20):
   write(out/'result.json',dict(status='minimum_certified',molecule=name,state=sid,max_gradient=gmax,frequencies_cm1=freq,energy_lowering_hartree=reference.energy_hartree-initial_energy,model_fingerprint=model.fingerprint,stages=stage+1,analytic_hessian=True,seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));return
 raise RuntimeError('Three stationary descendants retain negative nuclear curvature; all checkpoints retained')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('name');a.add_argument('sid');args=a.parse_args();lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 try:
  with threadpool_limits():run(args.name,args.sid)
 except Exception:
  write(OUT/f'geometry_escape/{args.name}/attempt_closed.json',dict(status='unresolved',traceback=traceback.format_exc(),source_sha256=sha(__file__),checkpoints_retained=True));raise
