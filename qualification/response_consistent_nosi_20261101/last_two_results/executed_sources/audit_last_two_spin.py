"""Independently validate spin-pure determinant sums for portable new cases."""
from pathlib import Path
import sys,socket,runpy,json,importlib.abc
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
ROOT=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-response-consistent-nosi-20261101')
class NoDFT(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='pyscf' or fullname.startswith('pyscf.'):raise AssertionError('DFT import in portable spin audit')
sys.meta_path.insert(0,NoDFT())
runpy.run_path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-orbital-performance-20261031/multicsf_integration/source/qualification/remote_bootstrap.py')
from matrix_gfn2.nosi_surface import PreparedNOSIModel,load_nosi_result
records=[]
for row in json.loads((ROOT/'last_two_protocol.json').read_text())['cases']:
 name=row['molecule'];model=PreparedNOSIModel(ROOT/'prepared_models/selected_cid/last_two'/name/'model.json')
 result=load_nosi_result(ROOT/'portable_validation/last_two'/name/'checkpoint.npz',model)
 states=model.apoc_states(result)
 assert len(states)==1 and states[0].spin_squared==0
 records.append(dict(id=row['id'],molecule=name,model_fingerprint=model.fingerprint,spin_squared=states[0].spin_squared,determinants=len(states[0].determinants)))
(ROOT/'last_two_spin_audit.json').write_text(json.dumps(dict(host=socket.getfqdn(),DFT_imports=0,new_SCC_calls=0,records=records),indent=2)+'\n')
print(json.dumps(records,indent=2),flush=True)
