"""Repair validation on the prior retained HNO point, including old failed FD steps."""
import importlib.util,json
from pathlib import Path
from types import SimpleNamespace
BASE=Path('/home/enzo/MOUSE-hno-fock-20261030')
spec=importlib.util.spec_from_file_location('p',BASE/'source/qualification/probe.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
np=p.np
from matrix_link.qmqm_component import CappedQMComponent
from matrix_gfn2 import load_response_reference,evaluate_composite_roks_analytic_hessian
with p.threadpool_limits(limits=2):
 calc=p.GFN2Calculator(p.p['atoms'],temperature_kelvin=0.,maximum_scf_iterations=500,scf_energy_tolerance_hartree=1e-10,scf_wavefunction_tolerance_electron=3.0000000000000004e-7,scf_mixer_tolerance=3.0000000000000004e-7,density_solver='diagonalization')
 component=CappedQMComponent(p.b,SimpleNamespace(_calculator=calc),BASE/'derivative_component')
 component.reference=p.c.reference;component.anchor=p.c.anchor
 r=component.reference[0];xyz=r.components[0].reference_state.coordinates_bohr/p.ANGSTROM_TO_BOHR
 refs=tuple(load_response_reference(path) for path in p.b.native_roks_composite_response_paths)
 h=evaluate_composite_roks_analytic_hessian((calc,calc),xyz,refs,reference_result=r)
 direction=-r.gradient_hartree_per_bohr;direction/=np.linalg.norm(direction)
 checks=[]
 for step in [2e-4,1e-4]:
  try:
   minus=component.evaluate(xyz-step/p.ANGSTROM_TO_BOHR*direction,compute_hessian=False)
   plus=component.evaluate(xyz+step/p.ANGSTROM_TO_BOHR*direction,compute_hessian=False)
   dg=(plus.gradient_hartree_per_bohr-minus.gradient_hartree_per_bohr)/(2*step)
   checks.append(dict(step_bohr=step,passed=True,maximum_hessian_vector_error=float(np.max(abs(dg-h.hessian_hartree_per_bohr2@direction.ravel()))),
    canonical_residuals=[z.execution['stationary_reference']['canonical_eigen_residuals_hartree'][-1] for z in [minus,plus]]))
  except Exception as exc:checks.append(dict(step_bohr=step,passed=False,error=str(exc)))
 row=dict(host=p.socket.getfqdn(),point='previous following endpoint, before additional geometry relaxation',checks=checks,
   component_csf_overlap=r.component_csf_overlap,weights=[.5,.5],state_energy_offset=0.)
 (BASE/'derivative_check.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
