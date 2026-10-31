"""Locate the strict canonical-reference closure floor without changing gates."""
import importlib.util,json
from pathlib import Path
BASE=Path('/home/enzo/MOUSE-roks-following-20261030')
spec=importlib.util.spec_from_file_location('campaign',BASE/'source/qualification/run_campaign.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
from matrix_gfn2.response_reference import stationary_response_reference
np=c.np
with c.threadpool_limits(limits=2):
 xyz0=np.array(json.loads((BASE/'results/nitroxyl/migration.json').read_text())['starting_coordinates_angstrom'])
 service,*_=c.service_for('nitroxyl','resume',xyz0);result=service._orbital_component.reference[0]
 state=result.components[0].reference_state;xyz=state.coordinates_bohr/c.ANGSTROM_TO_BOHR
 d=-result.gradient_hartree_per_bohr;d/=np.linalg.norm(d);xyz+=1e-3/c.ANGSTROM_TO_BOHR*d
 calc=service._matrix_gfn2_session._calculator
 try:ground,diagnostics=stationary_response_reference(calc,xyz)
 except RuntimeError as exc:
  ground=calc.evaluate(xyz,retain_orbitals=True,compute_cm5=False,compute_bond_orders=False);diagnostics=dict(error=str(exc))
 s=ground.orbital_state;k=calc.density_response_kernel(s);f=calc._model(s.coordinates_bohr,relocate_basis=True)[4]+k.density_dependent_fock(np.zeros_like(s.fock))
 sc=s.overlap@s.coefficients;canonical=(sc*s.orbital_energies_hartree)@sc.T
 diff=s.coefficients.T@(f-canonical)@s.coefficients
 n=int(np.sum(s.alpha_occupations))
 row=dict(host=c.HOST,diagnostics=diagnostics,actual_vs_exported_fock=float(np.max(abs(f-s.fock))),actual_vs_canonical=float(np.max(abs(f-canonical))),
  occupied_virtual_residual=float(np.max(abs(diff[:n,n:]))),occupied_occupied_residual=float(np.max(abs(diff[:n,:n]))),virtual_virtual_residual=float(np.max(abs(diff[n:,n:]))),orbital_energies=s.orbital_energies_hartree.tolist())
 c.write(BASE/'hno_reference_diagnostic.json',row);print(json.dumps(row))
