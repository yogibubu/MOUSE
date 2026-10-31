"""Remote preliminary assessment of a geometry-dependent ground replacement.
E=E0_native+0.5*sum(E_ROKS_i-E0_i); no fitted offset or changed weights.
"""
import importlib.util,json,time,sys
from pathlib import Path
BASE=Path('/home/enzo/MOUSE-roks-direct-20261030')
spec=importlib.util.spec_from_file_location('p',BASE/'source/qualification/probe.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
from matrix_gfn2.restricted_ground import evaluate_restricted_ground,restricted_ground_analytic_hessian
np=p.np
with p.threadpool_limits(limits=2):
 for name in sys.argv[1:]:
  start=time.monotonic();data,retained=p.retained(name);states=retained.reference[0].components
  xyz=states[0].reference_state.coordinates_bohr/p.ANGSTROM_TO_BOHR
  rows=[];grounds=[]
  for i,ref in enumerate([None]+[p.load_response_reference(x) for x in data['response_paths']]):
   calc=p.GFN2Calculator(data['atoms'],temperature_kelvin=0.)
   calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('unexpected ground SCC'))
   try:
    result=evaluate_restricted_ground(calc,xyz,template=states[0].reference_state,spin_reference=ref,gradient=True)
    grounds.append(result)
    row=dict(component=i,energy=result.energy_hartree,gradient=result.gradient_hartree_per_bohr.tolist(),residual=result.orbital_residual,minimum_curvature=result.minimum_orbital_curvature_hartree,iterations=result.iterations)
    if name=='nitroxyl':
     h=restricted_ground_analytic_hessian(calc,xyz,reference_result=result,spin_reference=ref)
     direction=-result.gradient_hartree_per_bohr;direction/=np.linalg.norm(direction)
     checks=[]
     for step in [2e-4,1e-4]:
      a=evaluate_restricted_ground(calc,xyz-step/p.ANGSTROM_TO_BOHR*direction,template=result.reference_state,spin_reference=ref,reference_result=result,gradient=True)
      b=evaluate_restricted_ground(calc,xyz+step/p.ANGSTROM_TO_BOHR*direction,template=result.reference_state,spin_reference=ref,reference_result=result,gradient=True)
      checks.append(dict(step=step,gradient_error=float(abs((b.energy_hartree-a.energy_hartree)/(2*step)-np.sum(result.gradient_hartree_per_bohr*direction))),hvp_error=float(np.max(abs((b.gradient_hartree_per_bohr-a.gradient_hartree_per_bohr).ravel()/(2*step)-h.hessian_hartree_per_bohr2@direction.ravel())))))
     row['derivative_checks']=checks
    rows.append(row);print(name,json.dumps(row),flush=True)
   except Exception as e:
    rows.append(dict(component=i,error=str(e)));print(name,'ERROR',str(e),flush=True);break
  out=dict(molecule=name,rows=rows,seconds=time.monotonic()-start)
  if len(grounds)==3:
   energy=grounds[0].energy_hartree+.5*sum(s.energy_hartree-g.energy_hartree for s,g in zip(states,grounds[1:]))
   gradient=grounds[0].gradient_hartree_per_bohr+.5*sum(s.gradient_hartree_per_bohr-g.gradient_hartree_per_bohr for s,g in zip(states,grounds[1:]))
   out.update(energy=energy,gradient=gradient.tolist(),excitation_ev=float((energy-grounds[0].energy_hartree)*27.211386245988),maximum_gradient=float(np.max(abs(gradient))),old_maximum_gradient=float(np.max(abs(retained.reference[0].gradient_hartree_per_bohr))))
  (BASE/'results'/name/'ground_replacement_probe.json').write_text(json.dumps(out,indent=2)+'\n')
