"""Checkpointed analytic nuclear Hessian columns; no electronic DFT backend."""
from pathlib import Path
import sys,json,socket,runpy,importlib.abc,hashlib,time,argparse,resource
assert socket.getfqdn()=='oracle.sns.it'
HERE=Path('/home/enzo/MOUSE-pi-pi-transfer-repair-20261101')
class NoDFT(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='pyscf' or fullname.startswith('pyscf.'):raise AssertionError('DFT backend forbidden')
sys.meta_path.insert(0,NoDFT());runpy.run_path('/home/enzo/MOUSE-orbital-performance-20261031/multicsf_integration/source/qualification/remote_bootstrap.py')
import numpy as np
from threadpoolctl import threadpool_limits
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from matrix_gfn2.nosi_surface import PreparedNOSIModel
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2._nosi_surface_response import MatrixResponse
from matrix_gfn2._nosi_nuclear import projected_eigen_jet
p=argparse.ArgumentParser();p.add_argument('cohort');p.add_argument('name');p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);a=p.parse_args()
with threadpool_limits(2):
 started=time.monotonic();model=PreparedNOSIModel(HERE/'prepared_models'/a.cohort/a.name/'model.json');symbols={1:'H',6:'C',7:'N',8:'O'};calc=GFN2Calculator([symbols[int(z)] for z in model.anchor.atomic_numbers],temperature_kelvin=0.);calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('SCC forbidden'))
 context=prepare_gfn2_deltascf(calc,model.anchor.coordinates_bohr/ANGSTROM_TO_BOHR,orbital_reference=model.anchor,spin_reference=model.response);arrays=model.arrays
 response=MatrixResponse(context,model.response,arrays['ground'],arrays['states'],arrays['multipliers'],arrays['auxiliary_frame'],model.labels,arrays['h'],arrays['eri'],auxiliary_records=model.auxiliary_records)
 destination=HERE/'compiled_validation'/a.cohort/a.name;destination.mkdir(parents=True,exist_ok=True);nr=model.anchor.coordinates_bohr.size
 source_files=[Path(__file__),Path(sys.modules['matrix_gfn2._nosi_surface_response'].__file__),Path(sys.modules['matrix_gfn2._nosi_configurations'].__file__),Path(sys.modules['matrix_gfn2._nosi_canonical'].__file__)]
 hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files};identity=dict(model=model.fingerprint,sources=hashes)
 rows=json.loads((HERE/'even_sector_5'/a.cohort/a.name/'result.json').read_text())['primary']['predictions']
 for column in range(a.worker,nr,a.workers):
  path=destination/f'column_{column:03d}.npz';metadata=path.with_suffix('.json')
  if path.exists() and metadata.exists():
   assert json.loads(metadata.read_text())['identity']==identity,'preserve changed-generation column';continue
  start=time.monotonic();h,s,eg=response.matrices(np.eye(nr)[column]);result={}
  for row in rows:
   jet=projected_eigen_jet(h,s,row['root']);gap=float(jet.v-eg.v)*27.211386245988;assert abs(gap-row['gap_ev'])<1e-5,(row['id'],gap,row['gap_ev'])
   result[row['id']+'_gradient']=jet.a;result[row['id']+'_hessian_column']=jet.ab;result[row['id']+'_energy']=jet.v
  temporary=path.with_suffix('.tmp.npz');np.savez_compressed(temporary,ground_gradient=eg.a,ground_hessian_column=eg.ab,**result);temporary.replace(path)
  metadata.write_text(json.dumps(dict(identity=identity,column=column,wall_seconds=time.monotonic()-start,worker_elapsed=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),indent=2)+'\n');print(a.name,column,'saved',flush=True)
