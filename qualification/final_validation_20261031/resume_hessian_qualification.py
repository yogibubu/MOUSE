"""Resume retained Hessian checks with controlled finite-difference convergence."""
from common import *
import fcntl
lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
out=OUT/'hessian_acceleration/qualification';model=MultiCSFModel(W/'models/Phthalimide/B024/model.json');calc=calculator(model.anchor)
calc.evaluate=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Unexpected SCC'))
source=Path(__import__('matrix_gfn2.nonlocal_response',fromlist=['x']).__file__).parent
hashes={p:sha(source/p) for p in ['nonlocal_response.py','local_frame_response.py','deltascf_analytic_hessian.py','multicsf_nuclear.py','restricted_ground.py','multicsf_surface.py']}
with threadpool_limits():
 fresh=load_multicsf_result(out/'Phthalimide/result.json',model,calc);hh=fresh.hessian_hartree_per_bohr2;g=fresh.gradient_hartree_per_bohr.ravel();x=fresh.reference_state.coordinates_bohr
 _,vec=np.linalg.eigh(hh);direction=vec[:,0];analytic=hh@direction;estimates=[];rows=[]
 for h in [2e-4,1e-4,5e-5]:
  points=[]
  for sign in [-1,1]:
   dest=(out/f'Phthalimide_direction_0_{sign:+d}') if h==2e-4 else out/f'Phthalimide_step_{h:g}_{sign:+d}'
   if (dest/'result.json').exists():trial=load_multicsf_result(dest/'result.json',model,calc)
   else:
    trial=evaluate_multicsf_surface(calc,(x+sign*h*direction.reshape(-1,3))/ANGSTROM_TO_BOHR,model,reference_result=fresh,compute_hessian=False);save_multicsf_result(dest,trial,model)
   points.append(trial)
  fd=(points[1].gradient_hartree_per_bohr.ravel()-points[0].gradient_hartree_per_bohr.ravel())/(2*h);estimates.append(fd)
  row=dict(step_bohr=h,central_error=float(np.max(abs(fd-analytic))),energy_gradient_error=float(abs((points[1].energy_hartree-points[0].energy_hartree)/(2*h)-g@direction)))
  if len(estimates)>1:
   extrapolated=(4*fd-estimates[-2])/3;row['richardson_error']=float(np.max(abs(extrapolated-analytic)));row['central_error_ratio']=rows[-1]['central_error']/row['central_error']
  rows.append(row);write(out/'phthalimide_step_convergence.json',rows);print('STEP',row,flush=True)
  if row.get('richardson_error',float('inf'))<2e-6:break
 else:raise AssertionError('Analytic Hessian not confirmed by controlled step convergence')
 assert all(r['energy_gradient_error']<2e-7 for r in rows)
 tr=[];rot=[]
 for axis in np.eye(3):
  tr.append(float(np.max(abs(hh@np.tile(axis,len(x))))));rot.append(float(np.max(abs(hh@np.cross(axis,x).ravel()-np.cross(axis,g.reshape(-1,3)).ravel()))))
 assert max(tr)<1e-8 and max(rot)<1e-7
 write(out/'phthalimide_validation.json',dict(status='pass',step_convergence=rows,translation_residual=max(tr),rotation_covariance_residual=max(rot),hessian_shape=hh.shape,source_sha256=hashes,finite_difference_hessian_used_in_production=False))
 comparisons=[]
 for name,sid,oldpath in [('Phthalimide','B024',OUT/'hessian_acceleration/Phthalimide/accepted/result.json'),('Furan','B022',OUT/'geometry/Furan/analytic_stationary/result.json'),('Cyclopentadiene','B016',OUT/'geometry_escape/Cyclopentadiene/stage_0/analytic_stationary/result.json')]:
  model=MultiCSFModel(W/f'models/{name}/{sid}/model.json');calc=calculator(model.anchor);calc.evaluate=lambda *a,**k:(_ for _ in ()).throw(AssertionError('Unexpected SCC'))
  old=load_multicsf_result(oldpath,model,calc);dest=out/name;start=time.monotonic()
  if (dest/'result.json').exists():fresh=load_multicsf_result(dest/'result.json',model,calc)
  else:
   fresh=evaluate_multicsf_surface(calc,old.reference_state.coordinates_bohr/ANGSTROM_TO_BOHR,model,reference_result=replace(old,hessian_hartree_per_bohr2=None),compute_hessian=True,follow_previous_csf=old.state_following=='local_csf');save_multicsf_result(dest,fresh,model)
  errors=dict(energy=abs(fresh.energy_hartree-old.energy_hartree),gradient=float(np.max(abs(fresh.gradient_hartree_per_bohr-old.gradient_hartree_per_bohr))),hessian=float(np.max(abs(fresh.hessian_hartree_per_bohr2-old.hessian_hartree_per_bohr2))))
  assert errors['energy']<1e-10 and errors['gradient']<2e-10 and errors['hessian']<3e-8,errors
  comparisons.append(dict(molecule=name,errors=errors,source_reference=str(oldpath),seconds_or_restore=time.monotonic()-start));print('MATRIX',comparisons[-1],flush=True)
 write(out/'result.json',dict(status='pass',whole_matrix_comparisons=comparisons,source_sha256=hashes,driver_sha256=sha(__file__)))
