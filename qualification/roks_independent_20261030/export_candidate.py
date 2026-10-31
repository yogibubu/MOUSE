"""Runnable no-offset LINK models and analytic Hessians for the independent set."""
from remote_bootstrap import HOST,BASE
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
import sys,json,time,traceback,numpy as np
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import state_from,write,key,digest
from matrix_gfn2 import GFN2Calculator,load_response_reference,evaluate_gfn2_roks_singlet,evaluate_gfn2_deltascf_analytic_hessian
from matrix_gfn2.deltascf_derivatives import save_derivative_point,load_derivative_point
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from matrix_link.scan import QMScanBackend
from pyscf import lib
lib.num_threads(2);ROOT=BASE/'results/candidate';ROOT.mkdir(exist_ok=True)
for path in sorted((BASE/'prepared').glob('*/case.json')):
 case=json.loads(path.read_text());name=case['name'];model=BASE/'results/models'/name/'shell_charge_spin.npz'
 if not model.exists():continue
 ref=load_response_reference(model)
 for target in case['targets']:
  state=target['id'];out=ROOT/name/state;out.mkdir(parents=True,exist_ok=True)
  try:
   rp=BASE/'results/predictions'/name/'shell_charge_spin'/state/'result.json'
   if not rp.exists():continue
   row=json.loads(rp.read_text())
   if not row['qualified']:continue
   old,xyz=state_from(target);s=old.reference_state
   r=load_derivative_point(row['checkpoint'],request_hash=row['request_hash'],coordinates_angstrom=xyz,require_gradient=False)
   calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
   ground=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=old.ground_energy_hartree)
   request=key(dict(model=digest(model),source=digest(row['checkpoint']),script=digest(__file__)))
   opts=dict(spin_reference=ref,orbital_solver='gmf',saddle_index=row['index'],orbital_tolerance=1e-10,maximum_iterations=120)
   cp=out/'central.npz'
   if cp.exists():central=load_derivative_point(cp,request_hash=request,coordinates_angstrom=xyz)
   else:
    central=evaluate_gfn2_roks_singlet(calc,xyz,*r.occupations,ground_state=ground,reference_result=r,gradient=True,**opts)
    save_derivative_point(cp,central,request_hash=request,coordinates_angstrom=xyz)
   hp=out/'analytic.npz'
   if hp.exists():
    with np.load(hp) as f:assert str(f['request_hash'])==request;hh=f['hessian'];curv=f['curvatures'];res=float(f['response_residual']);seconds=float(f['seconds'])
   else:
    start=time.perf_counter();h=evaluate_gfn2_deltascf_analytic_hessian(calc,xyz,*central.occupations,mode='roks',reference_result=central,spin_reference=ref)
    hh=h.hessian_hartree_per_bohr2;curv=np.linalg.eigvalsh(h.orbital_hessian_hartree);res=h.maximum_response_residual;seconds=time.perf_counter()-start
    np.savez_compressed(hp,request_hash=request,hessian=hh,gradient=central.gradient_hartree_per_bohr,curvatures=curv,response_residual=res,seconds=seconds)
   rng=np.random.default_rng(20261030);v=rng.normal(size=xyz.shape);v-=v.mean(axis=0);v/=np.linalg.norm(v);points=[];step=2e-4
   for sign in [-1,1]:
    geom=xyz+sign*step/ANGSTROM_TO_BOHR*v;p=out/f'validation_{sign:+d}.npz'
    if p.exists():point=load_derivative_point(p,request_hash=request,coordinates_angstrom=geom)
    else:
     point=evaluate_gfn2_roks_singlet(calc,geom,*central.occupations,reference_result=central,gradient=True,**opts)
     save_derivative_point(p,point,request_hash=request,coordinates_angstrom=geom)
    points.append(point)
   dg=(points[1].energy_hartree-points[0].energy_hartree)/(2*step)
   dh=(points[1].gradient_hartree_per_bohr-points[0].gradient_hartree_per_bohr).ravel()/(2*step)
   t=np.kron(np.ones((len(xyz),1)),np.eye(3));index=int(np.count_nonzero(curv < -1e-7))
   report=dict(host=HOST,molecule=name,state=state,model_fingerprint=ref.fingerprint,model_sha256=digest(model),request_hash=request,
    hessian_dimension=xyz.size,analytic_hessian_seconds=seconds,nuclear_displacements_in_analytic_method=0,
    gradient_direction_error=float(abs(dg-np.sum(central.gradient_hartree_per_bohr*v))),
    hessian_vector_error=float(np.max(abs(dh-hh@v.ravel()))),translation_sum_rule=float(np.max(abs(hh@t))),
    response_residual=res,gmf_index=central.orbital_saddle_index,analytic_orbital_index=index,
    energy_offset=0.,scope='absolute ROKS excited-state surface with frozen response; not DFT nuclear derivatives')
   report['passed']=bool(report['gradient_direction_error']<2e-5 and report['hessian_vector_error']<1e-4 and report['translation_sum_rule']<1e-7 and res<1e-8 and index==central.orbital_saddle_index)
   backend=QMScanBackend(name='matrix_gfn2',electronic_state=1,native_excited_method='roks',native_roks_response_path=model,
    deltascf_promotion=(target['hole'],target['particle']),orbital_saddle_index=index)
   write(out/'backend.json',asdict(backend));write(out/'validation.json',report);print('CANDIDATE',report,flush=True);assert report['passed']
  except Exception as exc:write(out/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()));print(traceback.format_exc(),flush=True)
