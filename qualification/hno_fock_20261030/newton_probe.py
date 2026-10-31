import importlib.util,json
from pathlib import Path
BASE=Path('/home/enzo/MOUSE-hno-fock-20261030')
spec=importlib.util.spec_from_file_location('p',BASE/'source/qualification/probe.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
np=p.np
from dataclasses import replace
from matrix_gfn2.ground_state_response import solve_ground_state_directional_response
from matrix_gfn2.response_reference import _reset_history
from scipy.linalg import eigh
with p.threadpool_limits(limits=2):
 calc=p.GFN2Calculator(p.p['atoms'],temperature_kelvin=0.)
 x=p.xyz+1e-3/p.ANGSTROM_TO_BOHR*p.v
 rows=[]
 for i in range(3):
  ground=calc.evaluate(x,retain_orbitals=True,compute_cm5=False,compute_bond_orders=False);s=ground.orbital_state;k=calc.density_response_kernel(s);core=calc._model(s.coordinates_bohr,relocate_basis=True)[4]
  sc=s.overlap@s.coefficients;fcan=(sc*s.orbital_energies_hartree)@sc.T;f=core+k.density_dependent_fock(np.zeros_like(s.fock));err=f-fcan
  response=solve_ground_state_directional_response(replace(s,fock=fcan),np.zeros_like(err),err,k,tolerance=1e-13)
  c=s.coefficients+response.coefficient_derivative
  vals,vec=np.linalg.eigh(c.T@s.overlap@c);c=c@((vec*vals**-.5)@vec.T)
  occ=s.alpha_occupations+s.beta_occupations;d0=(s.coefficients*occ)@s.coefficients.T;d1=(c*occ)@c.T
  f1=core+k.density_dependent_fock(d1-d0)
  eps=np.diag(c.T@f1@c);err1=np.max(abs(f1@c-s.overlap@c*eps))
  ne,nc=calc._active_native_context.generalized_eigh(f1)
  pe,pc=eigh(f1,s.overlap)
  rows.append(dict(correction=i,residual=float(np.max(abs(err))),response_residual=response.residual_norm,
   corrected_eigen_residual=float(err1),density_change=float(np.max(abs(d1-d0))),
   native_diagonalization_residual=float(np.max(abs(f1@nc-s.overlap@nc*ne))),
   scipy_diagonalization_residual=float(np.max(abs(f1@pc-s.overlap@pc*pe))),
   previous_seed_distance=None if i==0 else float(np.max(abs(prior-d0))),iterations=ground.iterations,scf_trace=calc._last_scf_trace))
  _reset_history(calc,cold=False);calc._previous_density=d1
  vals,vec=np.linalg.eigh(s.overlap);half=(vec*np.sqrt(vals))@vec.T
  calc._previous_orthogonal_density=half@d1@half;calc._previous_coordinates_bohr=s.coordinates_bohr.copy()
  calc._previous_orbital_energies=s.orbital_energies_hartree+response.orbital_energy_derivative;calc._previous_occupations=occ.copy();prior=d1
 (BASE/'newton_probe.json').write_text(json.dumps(rows,indent=2)+'\n')
 for row in rows:print({k:v for k,v in row.items() if k!='scf_trace'},flush=True)
