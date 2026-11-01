"""Check the analytic gradient against independent off-anchor scalar energies."""
from pathlib import Path
import socket,runpy,sys,json,time,shutil
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
BASE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101');HERE=BASE/'oracle/MOUSE-closing-validation-20261101';REPAIR=BASE/'oracle/MOUSE-pi-pi-transfer-repair-20261101';runpy.run_path(str(BASE/'source/qualification/remote_bootstrap.py'))
import numpy as np
from threadpoolctl import threadpool_limits
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.nosi_surface import PreparedNOSIModel,load_nosi_result
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
with threadpool_limits(2):
 folder=HERE/'off_anchor';folder.mkdir(exist_ok=True);shutil.copy2(HERE/'structures/acetaldehyde/accepted.npz',folder/'checkpoint.npz');model=PreparedNOSIModel(REPAIR/'prepared_models/extra_npi/acetaldehyde/model.json');r=load_nosi_result(folder/'checkpoint.npz',model);state=r.states[0];g=state.gradient_hartree_per_bohr;direction=g/np.linalg.norm(g);calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in model.anchor.atomic_numbers],temperature_kelvin=0.);rows=[]
 for step in [2e-4,1e-4]:
  e=[]
  for sign in [-1,1]:
   value=model.evaluate(calc,r.coordinates_angstrom+sign*step*direction/ANGSTROM_TO_BOHR,properties=('energy',),reference_result=r);e.append(value.states[0].energy_hartree)
  finite=(e[1]-e[0])/(2*step);analytic=float(np.sum(g*direction));rows.append(dict(step_bohr=step,finite=finite,analytic=analytic,error=abs(finite-analytic)))
 report=dict(rows=rows,coordinates_angstrom=r.coordinates_angstrom.tolist(),model=model.fingerprint,maximum_stationary_residual=r.maximum_stationary_residual);(folder/'gradient_check.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));assert max(row['error'] for row in rows)<1e-6
