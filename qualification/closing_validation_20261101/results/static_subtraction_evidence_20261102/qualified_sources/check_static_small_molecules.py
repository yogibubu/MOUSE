"""Revalidate the experimental subtraction against every frozen small model."""
from pathlib import Path
import socket,sys,runpy,importlib.abc,json,time,argparse
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
class NoDFT(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='pyscf' or fullname.startswith('pyscf.'):raise AssertionError('DFT runtime import')
sys.meta_path.insert(0,NoDFT())
BASE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101');HERE=BASE/'oracle/MOUSE-closing-validation-20261101';runpy.run_path(str(BASE/'source/qualification/remote_bootstrap.py'))
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.nosi_surface import PreparedNOSIModel,MatrixResponse
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from matrix_gfn2._nosi_static_subtraction import static_subtraction
from threadpoolctl import threadpool_limits
variant='disentangled_bath_spectral_block_reference_eta_0.600000';out=HERE/'static_subtraction_small_molecules';out.mkdir(exist_ok=True)
parser=argparse.ArgumentParser();parser.add_argument('--retry-rejected',action='store_true');args=parser.parse_args()
original=MatrixResponse.matrices;captured={}
def capture(self,direction):
 result=original(self,direction);captured['pencil']=result;return result
MatrixResponse.matrices=capture
rows=[]
with threadpool_limits(2):
 for path in sorted((HERE/variant).glob('*/*/model.json')):
  cohort,name=path.parent.parent.name,path.parent.name;destination=out/f'{cohort}_{name}.json'
  if destination.exists():
   saved=json.loads(destination.read_text())
   if not args.retry_rejected or saved['status']=='solved':rows.append(saved);continue
   destination.rename(destination.with_suffix('.rejected_initial.json'))
  start=time.monotonic()
  try:
   model=PreparedNOSIModel(path);calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in model.anchor.atomic_numbers],temperature_kelvin=0.);calc.evaluate=lambda *a,**k:(_ for _ in ()).throw(AssertionError('new SCC forbidden'))
   native=model.evaluate(calc,model.anchor.coordinates_bohr/ANGSTROM_TO_BOHR,properties=('energy',));hj,sj,egj=captured.pop('pencil');h=hj.v;s=sj.v;eg=float(egj.v)
   projection=np.eye(len(s))[:,1:];projection[0]-=s[0,1:]/s[0,0];sp=projection.T@s@projection;hp=projection.T@(h-eg*s)@projection
   nprimary=2*len(model.state_ids);r,delta,certificate=static_subtraction(hp,sp,np.arange(nprimary));e,z=eigh(hp+r@delta@r.T,sp);ci=projection@z
   overlaps=abs(native.ci_coefficients.T@s@ci);ii,jj=linear_sum_assignment(-overlaps);states=[]
   for i,j in zip(ii,jj):
    states.append(dict(id=native.states[i].state_id,native_gap_ev=native.states[i].excitation_hartree*27.211386245988,candidate_gap_ev=float(e[j]*27.211386245988),overlap=float(overlaps[i,j]),root=int(j+1)))
   np.savez_compressed(out/f'{cohort}_{name}.npz',h=h,s=s,ground_energy=eg,primary_counterterm=delta,energies=e,ci=ci)
   row=dict(cohort=cohort,molecule=name,status='solved',model_sha256=model.fingerprint,states=states,subtraction_certificate=certificate,wall_seconds=time.monotonic()-start)
  except Exception as error:row=dict(cohort=cohort,molecule=name,status='rejected',reason=repr(error),wall_seconds=time.monotonic()-start)
  destination.write_text(json.dumps(row,indent=2)+'\n');rows.append(row);print(json.dumps(row),flush=True)
(out/'summary.json').write_text(json.dumps(dict(scope='Experimental energy-only diagnostic; all rejected cases retained; no production promotion',host=socket.getfqdn(),rows=rows,DFT_imports=0,new_SCC_calls=0,TDA_solves=0),indent=2)+'\n')
