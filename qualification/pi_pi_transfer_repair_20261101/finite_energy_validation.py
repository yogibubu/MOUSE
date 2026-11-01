"""Independently reconverged energy differences; never an implemented derivative."""
from pathlib import Path
import sys,json,socket,runpy,importlib.abc,argparse,hashlib,time
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
p=argparse.ArgumentParser();p.add_argument('cohort');p.add_argument('name');a=p.parse_args()
with threadpool_limits(2):
 model=PreparedNOSIModel(HERE/'prepared_models'/a.cohort/a.name/'model.json');symbols={1:'H',6:'C',7:'N',8:'O'};calc=GFN2Calculator([symbols[int(z)] for z in model.anchor.atomic_numbers],temperature_kelvin=0.);calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('SCC forbidden'))
 xyz=model.anchor.coordinates_bohr/ANGSTROM_TO_BOHR;center=model.evaluate(calc,xyz,properties=('energy',));nr=xyz.size
 normal=np.array(model.manifest['provenance']['reflection_certificate']['normal']);out_of_plane=np.zeros_like(xyz);out_of_plane[0]=normal
 mixed=np.sin(np.arange(nr)+.37).reshape(xyz.shape);mixed-=mixed.mean(0);mixed/=np.linalg.norm(mixed);vectors=[out_of_plane.ravel(),mixed.ravel()]
 destination=HERE/'finite_energy'/a.cohort/a.name;destination.mkdir(parents=True,exist_ok=True)
 sources={name:hashlib.sha256(Path(sys.modules[name].__file__).read_bytes()).hexdigest() for name in ['matrix_gfn2.nosi_surface','matrix_gfn2._nosi_surface_response','matrix_gfn2._nosi_configurations','matrix_gfn2._nosi_canonical']};identity=dict(model=model.fingerprint,sources=sources)
 (destination/'center.json').write_text(json.dumps(dict(identity=identity,energies={s.state_id:s.energy_hartree for s in center.states},directions=[v.tolist() for v in vectors]),indent=2)+'\n')
 for i,direction in enumerate(vectors):
  for eps in [1e-3,5e-4]:
   for sign in [-1,1]:
    path=destination/f'direction_{i}_eps_{eps:.8f}_sign_{sign}.json'
    if path.exists():assert json.loads(path.read_text())['identity']==identity;continue
    start=time.monotonic();r=model.evaluate(calc,xyz+sign*eps*direction.reshape(xyz.shape)/ANGSTROM_TO_BOHR,properties=('energy',),reference_result=center)
    path.write_text(json.dumps(dict(identity=identity,energies={s.state_id:s.energy_hartree for s in r.states},maximum_stationary_residual=r.maximum_stationary_residual,minimum_overlap=min(s.overlap for s in r.states),wall_seconds=time.monotonic()-start),indent=2)+'\n');print(a.name,i,eps,sign,'saved',flush=True)
