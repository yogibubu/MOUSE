"""Remote analytic endpoint diagnostics; no claim of equilibrium frequencies."""
import importlib.util,sys,json,traceback
from pathlib import Path
BASE=Path('/home/enzo/MOUSE-hno-fock-20261030')
spec=importlib.util.spec_from_file_location('campaign',BASE/'source/qualification/resume_ground.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
from matrix_gfn2 import load_response_reference,evaluate_composite_roks_analytic_hessian
from matrix_gfn2.roks_tracking import active_orbital_character
from matrix_apoc import singlet_csf_overlap
np=c.np
with c.threadpool_limits(limits=2):
 for name in sys.argv[1:]:
  root=BASE/'results'/name
  initial=np.array(json.loads((root/'ground_migration.json').read_text())['starting_coordinates_angstrom'])
  service,_,_,backend,_=c.service_for(name,'ground_resume',initial);component=service._orbital_component
  result=component.reference[0];xyz=result.components[0].reference_state.coordinates_bohr/c.ANGSTROM_TO_BOHR
  refs=tuple(load_response_reference(p) for p in backend.native_roks_composite_response_paths)
  record=dict(host=c.HOST,molecule=name,role='terminal accepted state; nonstationary',energy_hartree=result.energy_hartree,
    maximum_gradient=float(np.max(abs(result.gradient_hartree_per_bohr))),indices=list(result.orbital_saddle_indices),component_csf_overlap=result.component_csf_overlap,
    original_anchor_csf=[float(singlet_csf_overlap(a,b,component._cross_overlap(ar.reference_state,br.reference_state))[0]) for a,b,ar,br in zip(component.anchor[1],component.reference[1],component.anchor[0].components,result.components)],
    active_character=[active_orbital_character(a,b,r.anchor_coordinates_bohr) for a,b,r in zip(component.anchor[0].components,result.components,refs)])
  record['native_ground_frontier_gaps_hartree']=[float(z.reference_state.orbital_energies_hartree[int(np.sum(z.reference_state.alpha_occupations))]-z.reference_state.orbital_energies_hartree[int(np.sum(z.reference_state.alpha_occupations))-1]) for z in result.components]
  direction=-result.gradient_hartree_per_bohr;direction/=np.linalg.norm(direction)
  trials=[]
  for step in [1e-3,1e-4]:
   try:
    point=component.evaluate(xyz+step/c.ANGSTROM_TO_BOHR*direction,compute_hessian=False)
    trials.append(dict(step_bohr=step,passed=True,energy_change=point.energy_hartree-result.energy_hartree,execution=point.execution))
   except Exception as exc:trials.append(dict(step_bohr=step,passed=False,error_type=type(exc).__name__,error=str(exc)))
  record['descent_trials']=trials
  try:
   h=evaluate_composite_roks_analytic_hessian((service._matrix_gfn2_session._calculator,)*2,xyz,refs,reference_result=result)
   record.update(minimum_absolute_orbital_curvatures_hartree=[z.minimum_absolute_orbital_curvature_hartree for z in h.components],
    orbital_eigenvalues_hartree=[np.linalg.eigvalsh(z.orbital_hessian_hartree).tolist() for z in h.components],
    nonstationary_frequencies_cm=c._exact_cartesian_normal_modes(c.prepare(name)['atoms'],xyz,h.hessian_hartree_per_bohr2).frequencies_cm.tolist())
   np.savez_compressed(root/'terminal_audit.npz',coordinates_angstrom=xyz,gradient=result.gradient_hartree_per_bohr,hessian=h.hessian_hartree_per_bohr2)
   checks=[]
   for step in [2e-4,1e-4]:
    try:
     minus=component.evaluate(xyz-step/c.ANGSTROM_TO_BOHR*direction,compute_hessian=False)
     plus=component.evaluate(xyz+step/c.ANGSTROM_TO_BOHR*direction,compute_hessian=False)
     dg=(plus.gradient_hartree_per_bohr-minus.gradient_hartree_per_bohr)/(2*step)
     checks.append(dict(step_bohr=step,maximum_hessian_vector_error=float(np.max(abs(dg-h.hessian_hartree_per_bohr2@direction.ravel())))))
    except Exception as exc:checks.append(dict(step_bohr=step,error=str(exc)))
   record['hessian_vector_checks']=checks
  except Exception as exc:record['hessian_failure']=str(exc);record['traceback']=traceback.format_exc()
  c.write(root/'terminal_audit.json',record);print(json.dumps(record),flush=True)
