"""Independent analytic curvature check at the new symmetry-broken minimum."""
from common import *
from geometry import vibrations
import fcntl
lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
with threadpool_limits():
 model=MultiCSFModel(W/'models/Cyclopentadiene/B016/model.json');calc=calculator(model.anchor)
 calc.evaluate=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Unexpected SCC'))
 root=OUT/'geometry_escape/Cyclopentadiene/stage_0/analytic_stationary'
 point=load_multicsf_result(root/'result.json',model,calc);freq,modes,masses=vibrations(point)
 direction=modes[:,0]/np.repeat(np.sqrt(masses),3);direction/=np.linalg.norm(direction)
 h=2e-4;grad=[];xyz=point.reference_state.coordinates_bohr
 for sign in [-1,1]:
  trial=evaluate_multicsf_surface(calc,(xyz+sign*h*direction.reshape(-1,3))/ANGSTROM_TO_BOHR,model,reference_result=point,follow_previous_csf=True)
  save_multicsf_result(root/f'soft_mode_{sign:+d}',trial,model);grad.append(trial.gradient_hartree_per_bohr.ravel())
 analytical=point.hessian_hartree_per_bohr2@direction;numerical=(grad[1]-grad[0])/(2*h);error=float(np.max(abs(analytical-numerical)))
 write(root.parent.parent/'minimum_mode_check.json',dict(status='pass' if error<5e-6 else 'fail',mode_frequency_cm1=float(freq[0]),step_bohr=h,max_hessian_vector_error=error,model_fingerprint=model.fingerprint,source_sha256=sha(__file__)))
 print('CPD minimum mode Hessian error',error,flush=True);assert error<5e-6
