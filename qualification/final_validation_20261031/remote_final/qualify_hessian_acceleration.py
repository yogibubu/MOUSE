"""Whole-matrix equivalence and independent Phthalimide derivative checks."""
from common import *
import fcntl,resource
lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
out=OUT/'hessian_acceleration/qualification';out.mkdir(parents=True,exist_ok=True);rows=[]
source=Path(__import__('matrix_gfn2.nonlocal_response',fromlist=['x']).__file__).parent
hashes={p:sha(source/p) for p in ['nonlocal_response.py','local_frame_response.py','deltascf_analytic_hessian.py','multicsf_nuclear.py','restricted_ground.py','multicsf_surface.py']}
with threadpool_limits():
 for name,sid,oldpath in [('Phthalimide','B024',OUT/'hessian_acceleration/Phthalimide/accepted/result.json'),('Furan','B022',OUT/'geometry/Furan/analytic_stationary/result.json'),('Cyclopentadiene','B016',OUT/'geometry_escape/Cyclopentadiene/stage_0/analytic_stationary/result.json')]:
  model=MultiCSFModel(W/f'models/{name}/{sid}/model.json');calc=calculator(model.anchor);calc.evaluate=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Unexpected SCC'))
  old=load_multicsf_result(oldpath,model,calc);start=time.monotonic();xyz=old.reference_state.coordinates_bohr/ANGSTROM_TO_BOHR
  fresh=evaluate_multicsf_surface(calc,xyz,model,reference_result=replace(old,hessian_hartree_per_bohr2=None),compute_hessian=True,follow_previous_csf=old.state_following=='local_csf')
  dest=out/name
  if not dest.exists():save_multicsf_result(dest,fresh,model)
  errors=dict(energy=abs(fresh.energy_hartree-old.energy_hartree),gradient=float(np.max(abs(fresh.gradient_hartree_per_bohr-old.gradient_hartree_per_bohr))),hessian=float(np.max(abs(fresh.hessian_hartree_per_bohr2-old.hessian_hartree_per_bohr2))))
  assert errors['energy']<1e-10 and errors['gradient']<2e-10 and errors['hessian']<3e-8,errors
  rows.append(dict(molecule=name,seconds=time.monotonic()-start,errors=errors,source_reference=str(oldpath)));print('MATRIX',name,rows[-1],flush=True)
  if name!='Phthalimide':continue
  hh=fresh.hessian_hartree_per_bohr2;g=fresh.gradient_hartree_per_bohr.ravel();x=old.reference_state.coordinates_bohr
  w,v=np.linalg.eigh(hh);rng=np.random.default_rng(7543);d=rng.normal(size=len(g));d/=np.linalg.norm(d);directions=[v[:,0],d];checks=[]
  for k,d in enumerate(directions):
   eps=2e-4;points=[]
   for sign in [-1,1]:
    trial=evaluate_multicsf_surface(calc,(x+sign*eps*d.reshape(-1,3))/ANGSTROM_TO_BOHR,model,reference_result=fresh,compute_hessian=False)
    save_multicsf_result(out/f'Phthalimide_direction_{k}_{sign:+d}',trial,model);points.append(trial)
   fd=(points[1].gradient_hartree_per_bohr.ravel()-points[0].gradient_hartree_per_bohr.ravel())/(2*eps)
   ee=abs((points[1].energy_hartree-points[0].energy_hartree)/(2*eps)-g@d);he=float(np.max(abs(fd-hh@d)))
   checks.append(dict(direction=k,step_bohr=eps,energy_gradient_error=float(ee),hessian_vector_error=he));write(out/'phthalimide_direction_checks.json',checks);print('FD',checks[-1],flush=True);assert ee<2e-7 and he<2e-6
  translation=[];rotation=[]
  for axis in np.eye(3):
   translation.append(float(np.max(abs(hh@np.tile(axis,len(x))))))
   rotation.append(float(np.max(abs(hh@np.cross(axis,x).ravel()-np.cross(axis,g.reshape(-1,3)).ravel()))))
  assert max(translation)<1e-8 and max(rotation)<1e-7
  write(out/'phthalimide_validation.json',dict(status='pass',directions=checks,translation_residual=max(translation),rotation_covariance_residual=max(rotation),hessian_shape=hh.shape,source_sha256=hashes,finite_difference_hessian_used_in_production=False))
 write(out/'result.json',dict(status='pass',whole_matrix_comparisons=rows,source_sha256=hashes,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
