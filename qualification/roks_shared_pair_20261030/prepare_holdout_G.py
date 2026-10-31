"""Remote prospective four-molecule control; reuse existing DFT references.

Freeze method/source hashes before parent predictions. TBEs do not enter the
response calibration, orbital assignment, root selection or ensemble weights.
"""
import socket,os,sys,json,hashlib,ast
from pathlib import Path
BASE=Path('/home/enzo/MOUSE-roks-relative-pair-20261030')
OLD=Path('/home/enzo/MOUSE-roks-independent-20261030')
HOST=socket.getfqdn();assert HOST=='oracle.sns.it'
os.environ['MOUSE_VALIDATION_ROOT']=str(BASE)
sys.path.insert(0,str(BASE/'source/qualification'));import remote_bootstrap
sys.path.append(str(OLD))
import numpy as np
from scipy.linalg import solve_sylvester
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
import full_response as full
import full_fast
from pyscf import lib
lib.num_threads(2)
NAMES=['cyanoformaldehyde','cyclopropenone','pyridazine','imidazole']
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,indent=2,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else a.item())+'\n')
def key(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
p=BASE/'holdout_G_protocol.json'
protocol=dict(host=socket.getfqdn(),molecules=NAMES,weights=[.5,.5],lambda_exchange=.9,offset=0.,
 reference='existing ground DFT PBE0 density/Fock/response; excited TBEs reporting only',
 method='independently stationary pure-singlet parents, fixed relative squared active overlaps from the assigned vertical parents',
 vertical_preservation='original independent 50/50 solution is stationary with zero constraint multipliers; no new vertical adjustment',
 molecular_status='held out from nonlocal-model and relative-constraint development; prior native/shell predictions exist',
 parent_solver='archived target assignment, first stationary saddle passing CSF >=.7, index search 0..3; no TBE proximity',
 source_sha256={name:digest(BASE/'source/packages/matrix-gfn2/src/matrix_gfn2'/name) for name in ['roks.py','orbital_saddle.py','shared_pair_roks.py']},
 calibration_source_sha256=digest(OLD/'full_fast.py'),spectral_source_sha256=digest(OLD/'qualified_spectral_response.py'))
if p.exists():
 saved=json.loads(p.read_text())
 assert {k:v for k,v in saved.items() if k!='source_sha256'}=={k:v for k,v in protocol.items() if k!='source_sha256'}
 protocol=saved
else:write(p,protocol)
# Extract the unchanged calibration function without executing that old script's
# development-campaign top-level statements.
src=OLD/'qualified_spectral_response.py';node=next(n for n in ast.parse(src.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='calibrate')
namespace=dict(globals(),ROOT=BASE/'results/qualified_spectral_response')
namespace['ROOT'].mkdir(parents=True,exist_ok=True)
exec(compile(ast.Module(body=[node],type_ignores=[]),str(src),'exec'),namespace)
spectral=namespace['calibrate']
with threadpool_limits(limits=2):
 for name in NAMES:
  case=json.loads((OLD/'prepared'/name/'case.json').read_text());case['targets']=[t for t in case['targets'] if t['id']=='n_pi']
  if len(case['targets'])!=1:raise ValueError('one preassigned n-pi state required')
  for profile,calibrate in [('full_response',full_fast.calibrate),('qualified_spectral_response',spectral)]:
   full.ROOT=BASE/'results'/profile;full.ROOT.mkdir(parents=True,exist_ok=True);full.calibrate=calibrate
   full.run(case)
  print('PREPARED_G',name,flush=True)
