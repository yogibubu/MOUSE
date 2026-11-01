"""Fresh-process audit: no DFT imports, TDA reads or SCC calls at runtime."""
from pathlib import Path
import sys,importlib.abc,socket,subprocess,runpy,argparse,time,json,resource
assert socket.getfqdn()=='oracle.sns.it'
ROOT=Path('/home/enzo/MOUSE-pi-pi-transfer-repair-20261101')
parser=argparse.ArgumentParser();parser.add_argument('cohort');parser.add_argument('name');parser.add_argument('--hessian',action='store_true');parser.add_argument('--gradient',action='store_true');parser.add_argument('--spin',action='store_true');args=parser.parse_args()
class NoDFT(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
    if fullname=='pyscf' or fullname.startswith('pyscf.'):raise AssertionError('DFT backend imported during prepared MOUSE evaluation')
sys.meta_path.insert(0,NoDFT())
runpy.run_path('/home/enzo/MOUSE-orbital-performance-20261031/multicsf_integration/source/qualification/remote_bootstrap.py')
import numpy as np
from threadpoolctl import threadpool_limits
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from matrix_gfn2.nosi_surface import PreparedNOSIModel,save_nosi_result,load_nosi_result
model=PreparedNOSIModel(ROOT/'prepared_models'/args.cohort/args.name/'model.json');symbols={1:'H',6:'C',7:'N',8:'O'};calculator=GFN2Calculator([symbols[int(z)] for z in model.anchor.atomic_numbers],temperature_kelvin=0.)
calculator.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('new SCC calculation forbidden in prepared runtime'))
reads=[];original=np.load

def safe_load(path,*a,**kw):
 text=str(path)
 if 'backend_tda' in text or text.endswith('backend.chk'):raise AssertionError('TDA/backend reference input read during prepared runtime')
 reads.append(text);return original(path,*a,**kw)
np.load=safe_load
with threadpool_limits(2):
 start=time.monotonic();properties=('energy','gradient','hessian') if args.hessian else ('energy','gradient') if args.gradient else ('energy',)
 result=model.evaluate(calculator,model.anchor.coordinates_bohr/ANGSTROM_TO_BOHR,properties=properties);wall=time.monotonic()-start
 target=json.loads((ROOT/'even_sector_5'/args.cohort/args.name/'result.json').read_text());expected={r['id']:r['gap_ev'] for r in target['primary']['predictions']};errors={};derivative_errors={}
 for state in result.states:
    errors[state.state_id]=abs(state.excitation_hartree*27.211386245988-expected[state.state_id]);assert errors[state.state_id]<1e-5,errors
 destination=ROOT/'portable_validation'/args.cohort/args.name;destination.mkdir(parents=True,exist_ok=True);save_nosi_result(destination/'checkpoint.npz',result);restored=load_nosi_result(destination/'checkpoint.npz',model)
 restarted=model.evaluate(calculator,restored.coordinates_angstrom,properties=('energy',),reference_result=restored)
 restart_error=max(abs(a.energy_hartree-b.energy_hartree) for a,b in zip(result.states,restarted.states));assert restart_error<1e-9,restart_error
 spin_report=[]
 if args.spin:
  spin_states=model.apoc_states(result)
  spin_report=[dict(state_id=r.state_id,spin_squared=float(state.spin_squared)) for r,state in zip(result.states,spin_states)]
 report=dict(spin=spin_report,host=socket.getfqdn(),properties=properties,wall_seconds=wall,peak_rss_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,prediction_errors_ev=errors,derivative_errors=derivative_errors,restart_error_hartree=restart_error,DFT_imports=0,new_SCC_calls=0,TDA_input_reads=0,read_paths=sorted(set(reads)),model_fingerprint=model.fingerprint)
 (destination/('hessian.json' if args.hessian else 'gradient.json' if args.gradient else 'energy.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
