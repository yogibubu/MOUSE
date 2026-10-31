"""Full molecular tensor-response nuclear Hessians, independently checked."""
from remote_bootstrap import HOST,BASE
from pathlib import Path
from types import SimpleNamespace
import sys,json,time,traceback,numpy as np
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import plan,state_from,write,digest,key
from matrix_gfn2 import GFN2Calculator,load_response_reference,evaluate_gfn2_roks_singlet,evaluate_gfn2_deltascf_analytic_hessian
from matrix_gfn2.deltascf_derivatives import load_derivative_point,save_derivative_point
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from pyscf import lib
lib.num_threads(2)
case=next(c for c in plan() if c['name']=='nitroxyl');target=case['targets'][0];r0,xyz=state_from(target);s=r0.reference_state
calc=GFN2Calculator([{1:'H',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
ground=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=r0.ground_energy_hartree)
for label in ['fock_shell_charge_spin']:
 out=BASE/'results/derivatives/nitroxyl'/label;out.mkdir(parents=True,exist_ok=True)
 try:
  model=BASE/'results/models/nitroxyl'/(label+'.npz');ref=load_response_reference(model)
  row=json.loads((BASE/'results/predictions/nitroxyl'/label/'n_pi/result.json').read_text());assert row['qualified']
  seed=load_derivative_point(row['checkpoint'],request_hash=row['request_hash'],coordinates_angstrom=xyz,require_gradient=False)
  request=key(dict(model=digest(model),script=digest(__file__)));opts=dict(spin_reference=ref,orbital_solver='gmf',saddle_index=row['index'],orbital_tolerance=1e-10,maximum_iterations=120)
  cp=out/'central.npz'
  if cp.exists():central=load_derivative_point(cp,request_hash=request,coordinates_angstrom=xyz)
  else:
   central=evaluate_gfn2_roks_singlet(calc,xyz,*seed.occupations,ground_state=ground,reference_result=seed,gradient=True,**opts)
   save_derivative_point(cp,central,request_hash=request,coordinates_angstrom=xyz)
  hp=out/'analytic.npz'
  if hp.exists():
   with np.load(hp) as f:assert str(f['request_hash'])==request;hh=f['hessian'];curv=f['curvatures'];res=float(f['response_error'])
  else:
   start=time.perf_counter();h=evaluate_gfn2_deltascf_analytic_hessian(calc,xyz,*seed.occupations,mode='roks',reference_result=central,spin_reference=ref)
   hh=h.hessian_hartree_per_bohr2;curv=np.linalg.eigvalsh(h.orbital_hessian_hartree);res=h.maximum_response_residual
   np.savez_compressed(hp,request_hash=request,hessian=hh,curvatures=curv,response_error=res,seconds=time.perf_counter()-start)
  numeric=np.empty_like(hh);energy_g=np.empty(xyz.size);step=2e-4
  for k in range(xyz.size):
   points=[]
   for sign in [-1,1]:
    geom=xyz.copy();geom.ravel()[k]+=sign*step/ANGSTROM_TO_BOHR;p=out/f'displaced_{k:02d}_{sign:+d}.npz'
    if p.exists():r=load_derivative_point(p,request_hash=request,coordinates_angstrom=geom)
    else:
     r=evaluate_gfn2_roks_singlet(calc,geom,*seed.occupations,reference_result=central,gradient=True,**opts)
     save_derivative_point(p,r,request_hash=request,coordinates_angstrom=geom)
    points.append(r)
   numeric[:,k]=(points[1].gradient_hartree_per_bohr-points[0].gradient_hartree_per_bohr).ravel()/(2*step)
   energy_g[k]=(points[1].energy_hartree-points[0].energy_hartree)/(2*step)
  t=np.kron(np.ones((len(xyz),1)),np.eye(3));translation=float(np.max(abs(hh@t)))
  report=dict(host=HOST,model=label,gradient_max_error=float(np.max(abs(energy_g-central.gradient_hartree_per_bohr.ravel()))),
   hessian_max_error=float(np.max(abs(numeric-hh))),translation_sum_rule=translation,response_residual=res,
   nuclear_displacements_in_analytic_method=0,validation_displacements=18,gmf_index=central.orbital_saddle_index,
   analytic_orbital_index=int(np.count_nonzero(curv < -1e-7)),scope='absolute excited-state surface; no energy offset; frozen response coefficients')
  report['passed']=bool(report['gradient_max_error']<2e-5 and report['hessian_max_error']<1e-4 and translation<1e-7 and res<1e-8 and report['gmf_index']==report['analytic_orbital_index'])
  write(out/'validation.json',report);np.savez_compressed(out/'numerical_validation.npz',gradient=energy_g,hessian=numeric);print(report,flush=True);assert report['passed']
 except Exception as exc:write(out/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()));print(traceback.format_exc(),flush=True)
