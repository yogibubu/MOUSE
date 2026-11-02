"""Remote cold-start, acceptance, rollback, coherent following and restart audit."""
from pathlib import Path
import runpy,socket,json,time,hashlib,sys,importlib.abc,argparse
class NoDFT(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='pyscf' or fullname.startswith('pyscf.'):raise AssertionError('DFT backend forbidden in LINK runtime')
sys.meta_path.insert(0,NoDFT())
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
runpy.run_path('/Users/vincenzobarone/MOUSE-oo-response-20261101/source/qualification/remote_bootstrap.py')
import numpy as np
from types import SimpleNamespace
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.nosi_surface import PreparedNOSIModel
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from matrix_link.scan import QMScanBackend
from matrix_link.qmqm_component import CappedQMComponent
ROOT=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-closing-validation-20261101')
parser=argparse.ArgumentParser();parser.add_argument('--variant',default='disentangled_bath_eta_0.600000');args=parser.parse_args()
p=ROOT/args.variant/'closing_blind/Ethylene/model.json'
m=PreparedNOSIModel(p);symbols={1:'H',6:'C',7:'N',8:'O'}
c=GFN2Calculator([symbols[int(z)] for z in m.anchor.atomic_numbers],temperature_kelvin=0.)
c.evaluate=lambda *a,**k:(_ for _ in ()).throw(AssertionError('unexpected SCC'))
b=QMScanBackend(name='matrix_gfn2',electronic_state=1,native_excited_method='nosi_50_50',native_nosi_model_path=p)
folder=ROOT/('link_'+args.variant)/time.strftime('%Y%m%d-%H%M%S')
component=CappedQMComponent(b,SimpleNamespace(_calculator=c),folder)
xyz=m.anchor.coordinates_bohr/ANGSTROM_TO_BOHR
start=time.monotonic();result=component.evaluate(xyz);component.accept(result)
manifest=folder/'nosi_accepted/accepted.json';before=manifest.read_bytes()
trial=xyz.copy();trial[0,0]+=.0001
step=component.evaluate(trial)
assert manifest.read_bytes()==before,'trial advanced accepted checkpoint'
restored=CappedQMComponent(b,SimpleNamespace(_calculator=c),folder)
repeat=restored.evaluate(xyz)
assert abs(result.energy_hartree-repeat.energy_hartree)<1e-9
assert repeat.execution['response_model']=='nosi_50_50_screened_even_cid'
assert repeat.execution.get('static_self_energy_subtraction',False)==m.manifest.get('static_self_energy_subtraction',False)
assert repeat.execution['spin_squared']==0 and step.execution['state_assignment_overlap']>.99
report=dict(static_self_energy_subtraction=repeat.execution.get('static_self_energy_subtraction',False),host=socket.getfqdn(),wall_seconds=time.monotonic()-start,cold_start_energy=result.energy_hartree,restart_error=abs(result.energy_hartree-repeat.energy_hartree),trial_overlap=step.execution['state_assignment_overlap'],trial_does_not_advance_accepted=True,spin_squared=0,new_SCC_calls=0,DFT_imports=0,selected_electronic_state=1,model_sha256=m.fingerprint)
(folder/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
