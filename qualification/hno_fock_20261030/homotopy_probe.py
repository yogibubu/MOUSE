import importlib.util,json
from pathlib import Path
BASE=Path('/home/enzo/MOUSE-hno-fock-20261030')
spec=importlib.util.spec_from_file_location('p',BASE/'source/qualification/probe.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
import matrix_gfn2.calculator as module
from matrix_gfn2.response_reference import stationary_response_reference
np=p.np
with p.threadpool_limits(limits=2):
 rows=[]
 for step in [0.,1e-4,2e-4,1e-3]:
  calc=p.GFN2Calculator(p.p['atoms'],temperature_kelvin=0.);evaluate=calc.evaluate;anneal=module._annealed_scf_temperature;count=[0]
  def correction(*a,**kw):
   count[0]+=1
   if count[0]>1:module._annealed_scf_temperature=lambda target,iteration,gap:target
   try:return evaluate(*a,**kw)
   finally:module._annealed_scf_temperature=anneal
  calc.evaluate=correction
  try:
   r,d=stationary_response_reference(calc,p.xyz+step/p.ANGSTROM_TO_BOHR*p.v)
   rows.append(dict(step_bohr=step,passed=True,diagnostics=d,energy=r.energy_hartree))
  except Exception as exc:rows.append(dict(step_bohr=step,passed=False,error=str(exc)))
  print(json.dumps(rows[-1]),flush=True)
 (BASE/'homotopy_probe.json').write_text(json.dumps(dict(host=p.socket.getfqdn(),rows=rows),indent=2)+'\n')
