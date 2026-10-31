"""Numerical checks of analytic curvatures at the new stationary geometries."""
from common import *
import fcntl
lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
with threadpool_limits():
 for name,sid in [('Furan','B022'),('Cyclopentadiene','B016')]:
  folder=OUT/f'geometry/{name}';model=MultiCSFModel(W/f'models/{name}/{sid}/model.json');calc=calculator(model.anchor);point=load_multicsf_result(folder/'analytic_stationary/result.json',model,calc)
  with np.load(folder/'vibrations.npz') as a:v=a['mass_weighted_modes'].copy();m=a['masses_amu'].copy();frequency=a['frequencies_cm1'].copy()
  step=2e-4;checks=[]
  for mode in [0,len(frequency)-1]:
   direction=v[:,mode]/np.repeat(np.sqrt(m),3);direction/=np.linalg.norm(direction);gradients=[]
   for sign in [-1,1]:
    cp=folder/f'mode_{mode}_{sign}.npz';x=point.reference_state.coordinates_bohr/ANGSTROM_TO_BOHR+sign*step*direction.reshape(-1,3)/ANGSTROM_TO_BOHR
    if cp.exists():
     with np.load(cp) as a:assert np.array_equal(x,a['coordinates']);gradient=a['gradient'].copy()
    else:
     q=evaluate_multicsf_surface(calc,x,model,reference_result=point,compute_hessian=False);gradient=q.gradient_hartree_per_bohr;np.savez_compressed(cp,coordinates=x,gradient=gradient,energy=q.energy_hartree,fidelity=q.fidelity,residual=q.orbital_residual)
    gradients.append(gradient.ravel())
   numerical=(gradients[1]-gradients[0])/(2*step);analytic=point.hessian_hartree_per_bohr2@direction;error=float(np.max(abs(numerical-analytic)));assert error<2e-5
   checks.append(dict(mode=mode,frequency_cm1=float(frequency[mode]),maximum_hessian_vector_error=error,analytic_curvature=float(direction@analytic),finite_difference_curvature=float(direction@numerical),step_bohr=step))
  write(folder/'stationary_mode_validation.json',dict(complete=True,host=HOST,source_sha256=sha(__file__),checks=checks,maximum_error=max(r['maximum_hessian_vector_error'] for r in checks),scope='most negative and highest positive vibrational directions at this stationary geometry; not a new full-column Hessian check'))
  print(name,'STATIONARY MODE CHECK',max(r['maximum_hessian_vector_error'] for r in checks),flush=True)
