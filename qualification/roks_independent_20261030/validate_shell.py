"""Evaluate the unchanged no-offset shell-response candidate on untouched molecules.

Tensor variants remain development diagnostics after the HNO failure.
"""
from remote_bootstrap import HOST,BASE
from pathlib import Path
import json,argparse,time,traceback
from types import SimpleNamespace
from calibrate import charge,state_from,GFN2Calculator,prepare_gfn2_deltascf,write,digest
from matrix_gfn2 import save_response_reference,load_response_reference
import evaluate
from pyscf import lib
lib.num_threads(8)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--group',type=int,default=0);p.add_argument('--groups',type=int,default=1);a=p.parse_args()
 cases=json.loads((BASE/'protocol.json').read_text())['cases']
 for i,c in enumerate(cases):
  if i%a.groups!=a.group:continue
  prepared=BASE/'prepared'/c['name']/'case.json'
  if not prepared.exists():print('NOT READY',c['name'],flush=True);continue
  case=json.loads(prepared.read_text())
  if not case['targets']:continue
  out=BASE/'results/models'/c['name'];out.mkdir(exist_ok=True)
  try:
   path=out/'shell_charge_spin.npz'
   if not path.exists():
    r,xyz=state_from(case['targets'][0]);s=r.reference_state
    calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
    ground=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=r.ground_energy_hartree)
    start=time.perf_counter();ref=charge.reference(case,prepare_gfn2_deltascf(calc,xyz,ground_state=ground))
    save_response_reference(path,ref)
    write(out/'shell_calibration.json',dict(host=HOST,seconds=time.perf_counter()-start,reference_sha256=digest(Path(case['reference'])/'reference.npz'),energy_offset=0.,profile='unchanged from prior cohort'))
   else:load_response_reference(path)
   evaluate.run(case)
  except Exception as exc:write(out/'validation_failure.json',dict(error=str(exc),traceback=traceback.format_exc()));print(traceback.format_exc(),flush=True)
