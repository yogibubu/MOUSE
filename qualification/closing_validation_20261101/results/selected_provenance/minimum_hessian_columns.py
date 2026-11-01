"""Checkpointed analytic columns at an optimized final-model minimum."""
from pathlib import Path
import socket,runpy,sys,json,argparse,time,hashlib,importlib.abc
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
class NoDFT(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='pyscf' or fullname.startswith('pyscf.'):raise AssertionError('DFT import forbidden at prepared minimum')
sys.meta_path.insert(0,NoDFT())
BASE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101');HERE=BASE/'oracle/MOUSE-closing-validation-20261101';REPAIR=BASE/'oracle/MOUSE-pi-pi-transfer-repair-20261101';runpy.run_path(str(BASE/'source/qualification/remote_bootstrap.py'))
import numpy as np
from threadpoolctl import threadpool_limits
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.nosi_surface import PreparedNOSIModel,load_nosi_result
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2._nosi_surface_response import MatrixResponse
from matrix_gfn2._nosi_nuclear import projected_eigen_jet
p=argparse.ArgumentParser();p.add_argument('cohort');p.add_argument('name');p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
with threadpool_limits(2):
 folder=HERE/'structures'/a.name;path=folder/'optimized.npz';model=PreparedNOSIModel(REPAIR/'prepared_models'/a.cohort/a.name/'model.json');r=load_nosi_result(path,model);xyz=r.coordinates_angstrom;nr=xyz.size;calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in model.anchor.atomic_numbers],temperature_kelvin=0.);calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('SCC forbidden'))
 context=prepare_gfn2_deltascf(calc,xyz,orbital_reference=model.anchor,spin_reference=model.response);data=model.arrays;response=MatrixResponse(context,model.response,r.ground_coefficients,r.state_coefficients,r.multipliers,data['auxiliary_frame'],model.labels,data['h'],data['eri'],auxiliary_records=model.auxiliary_records);destination=folder/'hessian_columns';destination.mkdir(exist_ok=True);identity=dict(model=model.fingerprint,optimized_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),response_sha256=hashlib.sha256(Path(sys.modules['matrix_gfn2._nosi_surface_response'].__file__).read_bytes()).hexdigest())
 for column in range(a.worker,nr,a.workers):
  target=destination/f'{column:03d}.npz';metadata=target.with_suffix('.json')
  if target.exists() and metadata.exists():assert json.loads(metadata.read_text())['identity']==identity;continue
  start=time.monotonic();h,s,eg=response.matrices(np.eye(nr)[column]);values={}
  for state in r.states:
   jet=projected_eigen_jet(h,s,state.root);assert abs(float(jet.v)-state.energy_hartree)<1e-9;assert np.max(abs(jet.a-state.gradient_hartree_per_bohr.ravel()))<1e-7
   values[state.state_id]=jet.ab
  temp=target.with_suffix('.tmp.npz');np.savez_compressed(temp,**values);temp.replace(target);metadata.write_text(json.dumps(dict(column=column,wall_seconds=time.monotonic()-start,identity=identity),indent=2)+'\n');print(a.name,column,'complete',flush=True)
