"""Fresh-process energy/spin/restart audit, with DFT/TDA/SCC forbidden."""
from pathlib import Path
import socket,sys,runpy,importlib.abc,argparse,json,time
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
class NoDFT(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='pyscf' or fullname.startswith('pyscf.'):raise AssertionError('DFT runtime import')
sys.meta_path.insert(0,NoDFT());BASE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101');HERE=BASE/'oracle/MOUSE-closing-validation-20261101';runpy.run_path(str(BASE/'source/qualification/remote_bootstrap.py'))
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.nosi_surface import PreparedNOSIModel,save_nosi_result,load_nosi_result
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from threadpoolctl import threadpool_limits
p=argparse.ArgumentParser();p.add_argument('cohort');p.add_argument('name');p.add_argument('--variant',default='disentangled_bath_eta_0.600000');a=p.parse_args();folder=HERE/a.variant/a.cohort/a.name;model=PreparedNOSIModel(folder/'model.json');target=json.loads((folder/'result.json').read_text());calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in model.anchor.atomic_numbers],temperature_kelvin=0.);calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('SCC runtime call'))
with threadpool_limits(2):
 start=time.monotonic();r=model.evaluate(calc,model.anchor.coordinates_bohr/ANGSTROM_TO_BOHR,properties=('energy',));seconds=time.monotonic()-start;save_nosi_result(folder/'runtime_checkpoint.npz',r);saved=load_nosi_result(folder/'runtime_checkpoint.npz',model);again=model.evaluate(calc,saved.coordinates_angstrom,properties=('energy',),reference_result=saved);spin=model.apoc_states(r);expected={s['id']:s['gap_ev'] for s in target['states']};errors={s.state_id:abs(s.excitation_hartree*27.211386245988-expected[s.state_id]) for s in r.states};restart=max(abs(s.energy_hartree-t.energy_hartree) for s,t in zip(r.states,again.states));spin_values=[float(s.spin_squared) for s in spin];assert max(errors.values())<1e-7 and restart<1e-9 and max(abs(s) for s in spin_values)<1e-8
 report=dict(host=socket.getfqdn(),model=model.fingerprint,wall_energy_seconds=seconds,energy_errors_ev=errors,restart_error_hartree=restart,spin_squared=spin_values,DFT_imports=0,new_SCC_calls=0,TDA_solves=0);(folder/'runtime_certificate.json').write_text(json.dumps(report,indent=2)+'\n');print(a.name,report,flush=True)
