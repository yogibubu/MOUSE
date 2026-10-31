"""Analytic full molecular Hessians and independent directional checks, uracil."""
from remote_bootstrap import HOST,BASE
from response_kernel_probe import reference,plan,state_from,write,key,digest,EV
from pathlib import Path
from types import SimpleNamespace
import numpy as np,json,time,traceback
from matrix_gfn2 import (GFN2Calculator,FrozenROKSAnchor,AnchoredROKSResult,evaluate_anchored_roks,
 evaluate_anchored_roks_analytic_hessian,save_anchored_roks_point,load_anchored_roks_point)
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.deltascf_derivatives import load_derivative_point
from pyscf import lib
lib.num_threads(2);ROOT=BASE/'results/derivatives/uracil';ROOT.mkdir(parents=True,exist_ok=True)
case=next(c for c in plan() if c['name']=='uracil');r0,xyz=state_from(case['targets'][0]);s=r0.reference_state
calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
ground=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=r0.ground_energy_hartree)
ref=reference(case,prepare_gfn2_deltascf(calc,xyz,ground_state=ground))
rows=json.loads((BASE/'results/response_kernel/uracil/result.json').read_text())['records']
for target in case['targets']:
 out=ROOT/target['id'];out.mkdir(exist_ok=True)
 try:
  row=next(x for x in rows if x['id']==target['id']);assert row['qualified']
  r=load_derivative_point(BASE/'results/response_kernel/uracil'/target['id']/'converged.npz',request_hash=row['request_hash'],coordinates_angstrom=xyz,require_gradient=False)
  dft=json.loads((BASE/'results/bhandhlyp/uracil'/(target['id']+'_frozen.json')).read_text());assert dft['qualified']
  anchor=FrozenROKSAnchor(ref,r.occupations,s.coordinates_bohr,dft['frozen_ev']/EV,row['frozen_ev']/EV,
   'BH&HLYP energy anchor '+digest(BASE/'results/bhandhlyp/uracil'/(target['id']+'_frozen.json')))
  request=key(dict(anchor=anchor.fingerprint,script=digest(__file__)))
  opts=dict(orbital_solver='gmf',saddle_index=r.orbital_saddle_index,orbital_tolerance=1e-10,maximum_iterations=100)
  path=out/'central.npz'
  if path.exists():central=load_anchored_roks_point(path,anchor,request_hash=request,coordinates_angstrom=xyz)
  else:
   central=evaluate_anchored_roks(calc,xyz,anchor,reference_result=AnchoredROKSResult(r,anchor),ground_state=ground,gradient=True,**opts)
   save_anchored_roks_point(path,central,request_hash=request,coordinates_angstrom=xyz)
  path=out/'analytic.npz'
  if path.exists():
   with np.load(path) as f:assert str(f['anchor'])==anchor.fingerprint;hessian=f['hessian'];curvatures=f['curvatures'];response_error=float(f['response_error'])
  else:
   start=time.perf_counter();h=evaluate_anchored_roks_analytic_hessian(calc,xyz,anchor,reference_result=central)
   hessian=h.hessian_hartree_per_bohr2;curvatures=np.linalg.eigvalsh(h.orbital_hessian_hartree);response_error=h.maximum_response_residual
   np.savez_compressed(path,anchor=anchor.fingerprint,hessian=hessian,gradient=central.gradient_hartree_per_bohr,
    curvatures=curvatures,response_error=response_error,seconds=time.perf_counter()-start)
  rng=np.random.default_rng(20261030);v=rng.normal(size=xyz.shape);v-=v.mean(axis=0);v/=np.linalg.norm(v)
  oxygen=int(np.flatnonzero(s.atomic_numbers==8)[0]);carbons=np.flatnonzero(s.atomic_numbers==6)
  carbon=int(carbons[np.argmin(np.linalg.norm(xyz[carbons]-xyz[oxygen],axis=1))])
  bond=xyz[oxygen]-xyz[carbon];bond/=np.linalg.norm(bond);stretch=np.zeros_like(xyz);stretch[oxygen]=bond;stretch[carbon]=-bond;stretch/=np.linalg.norm(stretch)
  checks=[];step=.001
  for i,direction in enumerate([v,stretch]):
   points=[]
   for sign in [-1,1]:
    geom=xyz+sign*step/ANGSTROM_TO_BOHR*direction;p=out/f'direction_{i}_{sign:+d}.npz'
    if p.exists():point=load_anchored_roks_point(p,anchor,request_hash=request,coordinates_angstrom=geom)
    else:
     point=evaluate_anchored_roks(calc,geom,anchor,reference_result=central,gradient=True,**opts)
     save_anchored_roks_point(p,point,request_hash=request,coordinates_angstrom=geom)
    points.append(point)
   numeric_g=(points[1].energy_hartree-points[0].energy_hartree)/(2*step)
   numeric_h=(points[1].gradient_hartree_per_bohr-points[0].gradient_hartree_per_bohr).ravel()/(2*step)
   check=dict(direction='random' if i==0 else 'carbonyl_stretch',gradient_error=abs(numeric_g-np.sum(central.gradient_hartree_per_bohr*direction)),
    hessian_vector_max_error=float(np.max(abs(numeric_h-hessian@direction.ravel()))),step_bohr=step)
   checks.append(check);print('DIRECTION',target['id'],check,flush=True)
  translations=np.kron(np.ones((len(xyz),1)),np.eye(3));terr=float(np.max(abs(hessian@translations)))
  index=int(np.count_nonzero(curvatures < -1e-7))
  report=dict(host=HOST,molecule='uracil',state=target['id'],anchor_fingerprint=anchor.fingerprint,
   nuclear_dimension=xyz.size,excitation_ev=central.excitation_energy_hartree*EV,checks=checks,
   translation_sum_rule=terr,orbital_response_residual=response_error,
   hessian_orbital_index=index,gmf_index=central.orbital_saddle_index,
   nuclear_displacements_used_in_analytic_method=0,scope='frozen molecular response and frozen DFT anchor')
  report['passed']=all(c['gradient_error']<2e-5 and c['hessian_vector_max_error']<1e-4 for c in checks) and terr<1e-7 and response_error<1e-8 and index==central.orbital_saddle_index
  write(out/'validation.json',report);print('VALIDATION',report,flush=True)
  assert report['passed']
 except Exception as exc:write(out/'failure.json',dict(host=HOST,error=str(exc),traceback=traceback.format_exc()));print(traceback.format_exc(),flush=True)
