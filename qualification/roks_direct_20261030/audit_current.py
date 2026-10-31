"""Independent energy/gradient/Hessian consistency at a retained direct endpoint."""
import importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
BASE=Path('/home/enzo/MOUSE-roks-direct-20261030')
spec=importlib.util.spec_from_file_location('p',BASE/'source/qualification/probe.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
from matrix_link.qmqm_component import CappedQMComponent
from matrix_link.scan import QMScanBackend
np=p.np
with p.threadpool_limits(limits=2):
 for name in sys.argv[1:]:
  data=json.loads((BASE/'results'/name/'prepared.json').read_text());calc=p.GFN2Calculator(data['atoms'],temperature_kelvin=0.)
  b=QMScanBackend(name='matrix_gfn2',electronic_state=1,native_excited_method='roks_50_50',deltascf_promotion=tuple(data['promotion']),native_roks_composite_response_paths=tuple(data['response_paths']),native_roks_composite_saddle_indices=tuple(data['saddle_indices']),native_roks_composite_following='local_csf_adaptive_index' if name=='nitroxyl' else 'local_csf',native_roks_composite_reference_policy='geometry_only',native_roks_composite_component_tracking='active_pair')
  c=CappedQMComponent(b,SimpleNamespace(_calculator=calc),BASE/'results'/name/'resume/roks_50_50')
  r=c.reference[0];xyz=r.components[0].reference_state.coordinates_bohr/p.ANGSTROM_TO_BOHR
  point=c.evaluate(xyz,compute_hessian=True);direction=-point.gradient_hartree_per_bohr.reshape(xyz.shape);direction/=np.linalg.norm(direction)
  checks=[]
  for step in [2e-4,1e-4,1e-5]:
   try:
    minus=c.evaluate(xyz-step/p.ANGSTROM_TO_BOHR*direction);plus=c.evaluate(xyz+step/p.ANGSTROM_TO_BOHR*direction)
    fd=(plus.energy_hartree-minus.energy_hartree)/(2*step)
    dg=(plus.gradient_hartree_per_bohr-minus.gradient_hartree_per_bohr)/(2*step)
    checks.append(dict(step_bohr=step,scalar_fd=fd,analytic=float(point.gradient_hartree_per_bohr@direction.ravel()),gradient_error=float(abs(fd-point.gradient_hartree_per_bohr@direction.ravel())),hvp_error=float(np.max(abs(dg-point.hessian_hartree_per_bohr2@direction.ravel())))))
   except Exception as exc:checks.append(dict(step_bohr=step,error=str(exc)))
  result=dict(molecule=name,maximum_gradient=float(np.max(abs(point.gradient_hartree_per_bohr))),energy=point.energy_hartree,checks=checks,execution=point.execution)
  (BASE/'results'/name/'audit_current.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(BASE/'results'/name/'audit_current.npz',xyz=xyz,hessian=point.hessian_hartree_per_bohr2,gradient=point.gradient_hartree_per_bohr)
  print(json.dumps(result),flush=True)
