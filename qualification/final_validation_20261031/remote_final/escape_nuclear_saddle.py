"""Bounded, checkpointed descent from negative curvature on the frozen PES."""
from common import *
from geometry import vibrations
import argparse,resource,traceback

def run(name,sid):
 model=MultiCSFModel(W/f'models/{name}/{sid}/model.json');calc=calculator(model.anchor)
 calc.evaluate=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Unexpected SCC in fixed surface'))
 source=OUT/f'geometry/{name}';out=OUT/f'geometry_escape/{name}';out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():return
 reference=replace(load_multicsf_result(source/'analytic_stationary/result.json',model,calc),state_following='local_csf');initial_energy=reference.energy_hartree;start=time.monotonic()
 for stage in range(3):
  dest=out/f'stage_{stage}';dest.mkdir(exist_ok=True);accepted=dest/'accepted.json';history=json.loads((dest/'history.json').read_text()) if (dest/'history.json').exists() else []
  eigenvalues,eigenvectors=eigh(reference.hessian_hartree_per_bohr2);inverse=(eigenvectors/np.maximum(abs(eigenvalues),.02))@eigenvectors.T
  if accepted.exists():
   reference=replace(load_multicsf_result(dest/json.loads(accepted.read_text())['path']/'result.json',model,calc),state_following='local_csf')
   if (dest/'inverse_bfgs.npz').exists():
    with np.load(dest/'inverse_bfgs.npz') as a:inverse=a['inverse'].copy()
  else:
   freq,modes,masses=vibrations(reference);direction=modes[:,0]/np.repeat(np.sqrt(masses),3);direction/=np.linalg.norm(direction);origin=reference.reference_state.coordinates_bohr.ravel()
   for amplitude in [.01,.005,.0025,.00125]:
    try:point=evaluate_multicsf_surface(calc,(origin+amplitude*direction).reshape(-1,3)/ANGSTROM_TO_BOHR,model,reference_result=reference,compute_hessian=False,follow_previous_csf=True)
    except (RuntimeError,ValueError,np.linalg.LinAlgError):continue
    if point.energy_hartree<reference.energy_hartree:break
   else:raise RuntimeError('No qualified downhill symmetry-breaking seed; original stationary checkpoint retained')
   reference=point;folder=dest/'accepted_000';save_multicsf_result(folder,point,model);write(accepted,dict(path=folder.name));history.append(dict(energy_hartree=point.energy_hartree,max_gradient=float(np.max(abs(point.gradient_hartree_per_bohr))),initial_displacement_bohr=amplitude));write(dest/'history.json',history)
  for iteration in range(max(1,100-len(history))):
   x=reference.reference_state.coordinates_bohr.ravel();g=reference.gradient_hartree_per_bohr.ravel();gmax=float(np.max(abs(g)))
   if gmax<3e-5:break
   direction=-inverse@g
   if direction@g>=0:inverse=np.eye(len(g));direction=-g
   direction*=min(1.,.04/max(np.linalg.norm(direction),1e-300));accepted_trial=False
   for fraction in [1.,.5,.25,.125,.0625,.03125,.015625,.0078125]:
    trial_x=x+fraction*direction
    try:point=evaluate_multicsf_surface(calc,trial_x.reshape(-1,3)/ANGSTROM_TO_BOHR,model,reference_result=reference,compute_hessian=False,follow_previous_csf=True)
    except (RuntimeError,ValueError,np.linalg.LinAlgError):continue
    if point.energy_hartree<=reference.energy_hartree+1e-4*fraction*(g@direction):accepted_trial=True;break
   if not accepted_trial:raise RuntimeError('Bounded nuclear line search reached state/continuation boundary; accepted point retained')
   dx=trial_x-x;dy=point.gradient_hartree_per_bohr.ravel()-g;curvature=dy@dx
   if curvature>1e-10 and curvature>1e-8*np.linalg.norm(dx)*np.linalg.norm(dy):
    rho=1/curvature;factor=np.eye(len(g))-rho*np.outer(dx,dy);inverse=factor@inverse@factor.T+rho*np.outer(dx,dx)
   reference=point;folder=dest/f'accepted_{len(history):03d}';save_multicsf_result(folder,point,model);write(accepted,dict(path=folder.name));np.savez_compressed(dest/'inverse_bfgs.tmp.npz',inverse=inverse);(dest/'inverse_bfgs.tmp.npz').replace(dest/'inverse_bfgs.npz')
   history.append(dict(energy_hartree=point.energy_hartree,max_gradient=float(np.max(abs(point.gradient_hartree_per_bohr))),fidelity=point.fidelity,step_bohr=float(np.linalg.norm(dx)),elapsed_seconds=time.monotonic()-start));write(dest/'history.json',history)
  if np.max(abs(reference.gradient_hartree_per_bohr))>=3e-5:raise RuntimeError('Bounded descent budget exhausted; accepted coordinates and inverse BFGS retained')
  folder=dest/'analytic_stationary'
  if (folder/'result.json').exists():reference=load_multicsf_result(folder/'result.json',model,calc)
  else:reference=evaluate_multicsf_surface(calc,reference.reference_state.coordinates_bohr/ANGSTROM_TO_BOHR,model,reference_result=reference,compute_hessian=True,follow_previous_csf=True);save_multicsf_result(folder,reference,model)
  freq,modes,masses=vibrations(reference);np.savez_compressed(dest/'vibrations.npz',frequencies_cm1=freq,mass_weighted_modes=modes,masses_amu=masses,coordinates_angstrom=reference.reference_state.coordinates_bohr/ANGSTROM_TO_BOHR)
  write(dest/'result.json',dict(max_gradient=float(np.max(abs(reference.gradient_hartree_per_bohr))),frequencies_cm1=freq,nuclear_minimum=bool(np.all(freq>=-20)),energy_hartree=reference.energy_hartree))
  if np.all(freq>=-20):
   write(out/'result.json',dict(status='minimum_certified',molecule=name,state=sid,max_gradient=float(np.max(abs(reference.gradient_hartree_per_bohr))),frequencies_cm1=freq,energy_lowering_hartree=reference.energy_hartree-initial_energy,model_fingerprint=model.fingerprint,stages=stage+1,analytic_hessian=True,seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256=sha(__file__)));return
 raise RuntimeError('Three stationary descendants retain negative nuclear curvature; accepted checkpoints retained')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('name');a.add_argument('sid');args=a.parse_args();from science_slots import science_locks;locks=science_locks()
 try:
  with threadpool_limits():run(args.name,args.sid)
 except Exception:write(OUT/f'geometry_escape/{args.name}/attempt_closed.json',dict(status='unresolved',traceback=traceback.format_exc(),source_sha256=sha(__file__),checkpoints_retained=True));raise
