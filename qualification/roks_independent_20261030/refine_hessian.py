"""Step refinement of the largest independent-set Hessian-vector discrepancy."""
from remote_bootstrap import HOST,BASE
from pathlib import Path
from types import SimpleNamespace
import sys,json,numpy as np
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import state_from,write,key,digest
from matrix_gfn2 import GFN2Calculator,load_response_reference,evaluate_gfn2_roks_singlet
from matrix_gfn2.deltascf_derivatives import load_derivative_point,save_derivative_point
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from pyscf import lib
lib.num_threads(2)
case=json.loads((BASE/'prepared/pyridazine/case.json').read_text());target=next(t for t in case['targets'] if t['id']=='pi_pi')
old,xyz=state_from(target);s=old.reference_state;out=BASE/'results/candidate/pyridazine/pi_pi'
meta=json.loads((out/'validation.json').read_text());ref=load_response_reference(BASE/'results/models/pyridazine/shell_charge_spin.npz')
central=load_derivative_point(out/'central.npz',request_hash=meta['request_hash'],coordinates_angstrom=xyz)
with np.load(out/'analytic.npz') as f:hessian=f['hessian']
calc=GFN2Calculator([{1:'H',6:'C',7:'N'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
rng=np.random.default_rng(20261030);v=rng.normal(size=xyz.shape);v-=v.mean(axis=0);v/=np.linalg.norm(v);rows=[]
for step in [1e-4,5e-5]:
 points=[]
 for sign in [-1,1]:
  geom=xyz+sign*step/ANGSTROM_TO_BOHR*v;p=out/f'refine_{step:.0e}_{sign:+d}.npz';request=key(dict(model=ref.fingerprint,step=step,version='refinement-v1'))
  if p.exists():point=load_derivative_point(p,request_hash=request,coordinates_angstrom=geom)
  else:
   point=evaluate_gfn2_roks_singlet(calc,geom,*central.occupations,reference_result=central,spin_reference=ref,gradient=True,
    orbital_solver='gmf',saddle_index=central.orbital_saddle_index,orbital_tolerance=1e-11,maximum_iterations=120)
   save_derivative_point(p,point,request_hash=request,coordinates_angstrom=geom)
  points.append(point)
 dg=(points[1].gradient_hartree_per_bohr-points[0].gradient_hartree_per_bohr).ravel()/(2*step)
 rows.append(dict(step_bohr=step,hessian_vector_error=float(np.max(abs(dg-hessian@v.ravel())))))
report=dict(host=HOST,original_step_bohr=2e-4,original_error=meta['hessian_vector_error'],refinement=rows,
 passed=bool(rows[-1]['hessian_vector_error']<3e-6 and rows[0]['hessian_vector_error']<meta['hessian_vector_error']))
write(out/'step_refinement.json',report);print(report);assert report['passed']
