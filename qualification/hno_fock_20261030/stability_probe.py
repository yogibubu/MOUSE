import importlib.util,json
from pathlib import Path
BASE=Path('/home/enzo/MOUSE-hno-fock-20261030')
spec=importlib.util.spec_from_file_location('p',BASE/'source/qualification/probe.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
from matrix_gfn2.response_reference import stationary_response_reference
np=p.np
with np.load(BASE/'results/nitroxyl/terminal_audit.npz') as d:xyz=d['coordinates_angstrom'].copy();v=-d['gradient'];v/=np.linalg.norm(v)
rows=[]
with p.threadpool_limits(limits=2):
 for step in [-2e-4,-1e-4,0.,1e-4,2e-4]:
  calc=p.GFN2Calculator(p.p['atoms'],temperature_kelvin=0.,maximum_scf_iterations=500,scf_energy_tolerance_hartree=1e-10,scf_wavefunction_tolerance_electron=3.0000000000000004e-7,scf_mixer_tolerance=3.0000000000000004e-7,density_solver='diagonalization')
  try:
   r,d=stationary_response_reference(calc,xyz+step/p.ANGSTROM_TO_BOHR*v)
   row=dict(step_bohr=step,passed=True,diagnostics=d)
  except Exception as exc:row=dict(step_bohr=step,passed=False,error=str(exc))
  rows.append(row);print(json.dumps(row),flush=True)
 (BASE/'stability_probe.json').write_text(json.dumps(dict(coordinates_angstrom=xyz.tolist(),direction=v.tolist(),rows=rows),indent=2)+'\n')
