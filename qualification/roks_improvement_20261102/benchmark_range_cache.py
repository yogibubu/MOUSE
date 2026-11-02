"""Matched action benchmark at one qualified molecular stationary point."""
import socket,runpy,json,time
from pathlib import Path
from types import MethodType
import numpy as np
from pyscf import dft,scf
from threadpoolctl import threadpool_limits
assert socket.getfqdn().lower()=='mac-2.homenet.telecomitalia.it'
runpy.run_path('/Users/vincenzobarone/MOUSE-oo-response-20261101/source/qualification/remote_bootstrap.py')
from matrix_pyscf._oo_surface import OrbitalSurface
r=Path(__file__).resolve().parent
mol=scf.chkfile.load_mol('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-cam-response-20261101/data/closing_blind/references/Butadiene/backend.chk')
with np.load(r/'vertical/B003/WB97X/qualified.npz') as z:c=z['coefficients'];occ=z['occupations']
rows={};actions={}
with threadpool_limits(2):
 for label in ['native','cached']:
  mf=dft.UKS(mol);mf.xc='WB97X';mf.grids.level=3;mf.small_rho_cutoff=0
  surface=OrbitalSurface(mf,c,occ,common=True)
  if label=='native':
   for w,m,o in surface.components:m.get_jk=MethodType(type(m).get_jk,m)
  start=time.perf_counter();data=surface.evaluate(c);elapsed=time.perf_counter()-start
  actions[label]=data[2];rows[label]=dict(setup_energy_gradient_response_seconds=elapsed,action_seconds=[],surface=surface)
 rng=np.random.default_rng(183);v=rng.normal(size=surface.dimension);v/=np.linalg.norm(v);values={}
 for i in range(3):
  for label in (['native','cached'] if i%2==0 else ['cached','native']):
   start=time.perf_counter();values[label]=actions[label](v);rows[label]['action_seconds'].append(time.perf_counter()-start)
 error=float(np.max(abs(values['native']-values['cached'])));assert error<1e-10,error
 for row in rows.values():
  row['statistics']=row.pop('surface').statistics;row['median_action_seconds']=float(np.median(row['action_seconds']))
report=dict(host=socket.getfqdn(),molecule='Butadiene',functional='WB97X',basis='aug-cc-pVDZ',threads=2,
    ground_SCF_calls=0,TDA_calls=0,maximum_action_difference=error,rows=rows,
    median_action_speedup=rows['native']['median_action_seconds']/rows['cached']['median_action_seconds'],
    scope='Same orbitals and operator; action/setup timings only, not complete calculation speedup or memory benchmark')
(r/'range_cache_benchmark.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
