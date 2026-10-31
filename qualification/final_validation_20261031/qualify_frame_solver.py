"""Compare the revised solver against independent archived displaced gradients."""
from common import *
import fcntl
from matrix_gfn2 import multicsf_frame
lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
rows=[]
with threadpool_limits():
 for name,sid in [('Furan','B022'),('Cyclopentadiene','B016')]:
  root=OUT/f'geometry/{name}';model=MultiCSFModel(W/f'models/{name}/{sid}/model.json');calc=calculator(model.anchor);point=load_multicsf_result(root/'analytic_stationary/result.json',model,calc)
  for sign in [-1,1]:
   p=root/f'mode_0_{sign}.npz'
   with np.load(p) as a:xyz=a['coordinates'].copy();gradient=a['gradient'].copy();energy=float(a['energy'])
   value=evaluate_multicsf_surface(calc,xyz,model,reference_result=point,compute_hessian=False);ge=float(np.max(abs(value.gradient_hartree_per_bohr-gradient)));ee=abs(value.energy_hartree-energy)
   assert ge<2e-7 and ee<1e-10 and value.orbital_residual<1e-9
   rows.append(dict(molecule=name,sign=sign,gradient_max_error=ge,energy_error=ee,orbital_residual=value.orbital_residual))
 write(OUT/'frame_solver_validation.json',dict(status='passed',core_solver_sha256=sha(multicsf_frame.__file__),checks=rows,scope='four independently archived off-anchor total-energy/gradient evaluations reproduced; Hessian energy definition unchanged'))
 print('FRAME SOLVER PASS',rows,flush=True)
