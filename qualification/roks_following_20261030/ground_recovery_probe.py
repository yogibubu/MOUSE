import importlib.util,json
from pathlib import Path
BASE=Path('/home/enzo/MOUSE-roks-following-20261030')
spec=importlib.util.spec_from_file_location('campaign',BASE/'source/qualification/run_campaign.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
from matrix_gfn2.response_reference import stationary_response_reference
np=c.np
rows=[]
with c.threadpool_limits(limits=2):
 for name in ['nitroxyl','acetaldehyde']:
  initial=np.array(json.loads((BASE/'results'/name/'migration.json').read_text())['starting_coordinates_angstrom'])
  service,*_=c.service_for(name,'resume',initial);r=service._orbital_component.reference[0];s=r.components[0].reference_state
  xyz=s.coordinates_bohr/c.ANGSTROM_TO_BOHR;direction=-r.gradient_hartree_per_bohr;direction/=np.linalg.norm(direction)
  calc=service._matrix_gfn2_session._calculator
  for step in [1e-3,2e-4,1e-4]:
   point=xyz+step/c.ANGSTROM_TO_BOHR*direction
   try:
    ground,diagnostics=stationary_response_reference(calc,point,seed_state=s)
    rows.append(dict(molecule=name,step_bohr=step,passed=True,diagnostics=diagnostics))
   except Exception as exc:rows.append(dict(molecule=name,step_bohr=step,passed=False,error=str(exc)))
  c.write(BASE/'ground_recovery_probe.json',dict(host=c.HOST,rows=rows))
  print(json.dumps(rows[-3:]),flush=True)
