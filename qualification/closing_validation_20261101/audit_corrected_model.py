"""Fresh-process analytic E/G/H and runtime-contract audit of a prepared model."""
from pathlib import Path
import sys,importlib.abc,socket,runpy,argparse,time,json,resource,hashlib
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
BASE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101');HERE=BASE/'oracle/MOUSE-closing-validation-20261101'
class NoDFT(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='pyscf' or fullname.startswith('pyscf.'):raise AssertionError('DFT imported in prepared runtime')
sys.meta_path.insert(0,NoDFT());runpy.run_path(str(BASE/'source/qualification/remote_bootstrap.py'))
import numpy as np
from threadpoolctl import threadpool_limits
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.nosi_surface import PreparedNOSIModel,save_nosi_result,load_nosi_result
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
parser=argparse.ArgumentParser();parser.add_argument('variant');parser.add_argument('cohort');parser.add_argument('name');args=parser.parse_args();root=HERE/args.variant/args.cohort/args.name;out=root/'analytic_audit';out.mkdir(exist_ok=True)
model=PreparedNOSIModel(root/'model.json');calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in model.anchor.atomic_numbers],temperature_kelvin=0.)
calc.evaluate=lambda *a,**k:(_ for _ in ()).throw(AssertionError('new SCC forbidden'))
original_load=np.load;read_paths=[]
def safe_load(path,*a,**k):
 text=str(path);assert 'backend_tda' not in text and not text.endswith('backend.chk'),text;read_paths.append(text);return original_load(path,*a,**k)
np.load=safe_load
with threadpool_limits(2):
 t=time.monotonic();xyz=model.anchor.coordinates_bohr/ANGSTROM_TO_BOHR
 if (out/'hessian.npz').exists():result=load_nosi_result(out/'hessian.npz',model);wall=None
 else:result=model.evaluate(calc,xyz,properties=('energy','gradient','hessian'));wall=time.monotonic()-t;save_nosi_result(out/'hessian.npz',result)
 spin=model.apoc_states(result);rows=[]
 n=xyz.size;directions=[]
 for k in range(2):
  vector=np.sin(np.arange(n)*(.73+k*.41)+.29).reshape(-1,3);vector-=vector.mean(0);directions.append(vector.ravel()/np.linalg.norm(vector))
 values={}
 for di,vector in enumerate(directions):
  for step in [1e-3,5e-4]:
   for sign in [-1,1]:
    file=out/f'finite_d{di}_h{step}_s{sign}.json'
    if file.exists():data=json.loads(file.read_text());assert data['model']==model.fingerprint
    else:
     v=model.evaluate(calc,xyz+sign*step*vector.reshape(-1,3)/ANGSTROM_TO_BOHR,properties=('energy',),reference_result=result);data=dict(model=model.fingerprint,energies={s.state_id:s.energy_hartree for s in v.states},overlaps={s.state_id:s.overlap for s in v.states});file.write_text(json.dumps(data,indent=2)+'\n')
    values[di,step,sign]=data
 for state,sp in zip(result.states,spin):
  checks=[]
  for di,vector in enumerate(directions):
   estimates=[]
   for step in [1e-3,5e-4]:
    em=values[di,step,-1]['energies'][state.state_id];ep=values[di,step,1]['energies'][state.state_id];estimates.append(((ep-em)/(2*step),(ep+em-2*state.energy_hartree)/step**2))
   rich=(4*np.array(estimates[1])-estimates[0])/3;analytic=[state.gradient_hartree_per_bohr.ravel()@vector,vector@state.hessian_hartree_per_bohr2@vector]
   checks.append(dict(direction=di,analytic=[float(x) for x in analytic],richardson=rich.tolist(),absolute_error=abs(rich-analytic).tolist(),steps_bohr=[1e-3,5e-4]))
  assert abs(float(sp.spin_squared))<1e-8
  rows.append(dict(id=state.state_id,gap_ev=state.excitation_hartree*27.211386245988,spin_squared=float(sp.spin_squared),hessian_asymmetry=float(np.max(abs(state.hessian_hartree_per_bohr2-state.hessian_hartree_per_bohr2.T))),checks=checks))
 restarted=model.evaluate(calc,xyz,properties=('energy',),reference_result=result);restart_error=max(abs(a.energy_hartree-b.energy_hartree) for a,b in zip(result.states,restarted.states))
 maxg=max(x['absolute_error'][0] for r in rows for x in r['checks']);maxh=max(x['absolute_error'][1] for r in rows for x in r['checks'])
 report=dict(model=model.fingerprint,host=socket.getfqdn(),variant=args.variant,molecule=args.name,wall_hessian_seconds=wall,peak_rss_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,states=rows,maximum_gradient_error=maxg,maximum_hessian_curvature_error=maxh,restart_error_hartree=restart_error,DFT_imports=0,SCC_calls=0,TDA_reads=0,read_paths=sorted(set(read_paths)),finite_differences='validation only; production E/G/H analytic',runtime_sources={name:hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest() for name,mod in sys.modules.items() if name.startswith('matrix_gfn2._nosi_') or name=='matrix_gfn2.nosi_surface'})
 (out/'certificate.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True);assert maxg<1e-7 and maxh<2e-6 and restart_error<1e-9
