"""Public-API spin, restart and optional analytic Hessian qualification."""
from pathlib import Path
import argparse,socket,sys,runpy,importlib.abc,json,time,shutil
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
class NoDFT(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='pyscf' or fullname.startswith('pyscf.'):raise AssertionError('DFT runtime import')
sys.meta_path.insert(0,NoDFT());BASE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101');HERE=BASE/'oracle/MOUSE-closing-validation-20261101';runpy.run_path(str(BASE/'source/qualification/remote_bootstrap.py'))
import numpy as np
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.nosi_surface import PreparedNOSIModel,save_nosi_result,load_nosi_result
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from threadpoolctl import threadpool_limits
def main():
 p=argparse.ArgumentParser();p.add_argument('cohort');p.add_argument('name');p.add_argument('--hessian',action='store_true');p.add_argument('--hessian-workers',type=int,default=1);a=p.parse_args()
 source=HERE/'disentangled_bath_spectral_block_reference_eta_0.600000'/a.cohort/a.name;out=HERE/'static_subtraction_models'/a.cohort/a.name;out.mkdir(parents=True,exist_ok=True)
 suffix='' if a.hessian_workers==1 else f'_parallel_{a.hessian_workers}'
 certificate_path=out/(f'hessian{suffix}_certificate.json' if a.hessian else 'runtime_certificate.json')
 if certificate_path.exists():print('REUSE',certificate_path,flush=True);raise SystemExit(0)
 manifest=json.loads((source/'model.json').read_text());manifest['static_self_energy_subtraction']=True;manifest['subtraction_scope']='Experimental general static matching; no fitted offset'
 for name in manifest['sha256']:
  if not (out/name).exists():shutil.copy2(source/name,out/name)
 (out/'model.json').write_text(json.dumps(manifest,indent=2)+'\n');model=PreparedNOSIModel(out/'model.json')
 calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in model.anchor.atomic_numbers],temperature_kelvin=0.);calc.evaluate=lambda *a,**k:(_ for _ in ()).throw(AssertionError('new SCC forbidden'))
 with threadpool_limits(2):
  xyz=model.anchor.coordinates_bohr/ANGSTROM_TO_BOHR;start=time.monotonic();result=model.evaluate(calc,xyz,properties=('energy','gradient','hessian') if a.hessian else ('energy',),hessian_workers=a.hessian_workers);seconds=time.monotonic()-start
  save_nosi_result(out/(f'hessian{suffix}.npz' if a.hessian else 'runtime_checkpoint.npz'),result)
  saved=load_nosi_result(out/(f'hessian{suffix}.npz' if a.hessian else 'runtime_checkpoint.npz'),model);again=model.evaluate(calc,xyz,properties=('energy',),reference_result=saved)
  restart=max(abs(s.energy_hartree-t.energy_hartree) for s,t in zip(result.states,again.states));assert restart<1e-9
  expected=json.loads((HERE/'static_subtraction_small_molecules'/f'{a.cohort}_{a.name}.json').read_text());expected={s['id']:s['candidate_gap_ev'] for s in expected['states']};energy_error=max(abs(s.excitation_hartree*27.211386245988-expected[s.state_id]) for s in result.states);assert energy_error<1e-7
  spin=[float(s.spin_squared) for s in model.apoc_states(result)];assert max(abs(x) for x in spin)<1e-8
  checks=[];serial_comparison=None
  if a.hessian and a.hessian_workers>1 and (out/'hessian.npz').exists():
   serial=load_nosi_result(out/'hessian.npz',model)
   serial_comparison=dict(energy_max_error_hartree=max(abs(x.energy_hartree-y.energy_hartree) for x,y in zip(result.states,serial.states)),gradient_max_error_hartree_per_bohr=max(float(np.max(abs(x.gradient_hartree_per_bohr-y.gradient_hartree_per_bohr))) for x,y in zip(result.states,serial.states)),Hessian_max_error_hartree_per_bohr2=max(float(np.max(abs(x.hessian_hartree_per_bohr2-y.hessian_hartree_per_bohr2))) for x,y in zip(result.states,serial.states)))
   assert max(serial_comparison.values())<1e-8,serial_comparison
  elif a.hessian:
   rng=np.random.default_rng(209);directions=[rng.normal(size=xyz.shape),np.zeros_like(xyz)];directions[1][:,2]=rng.normal(size=len(xyz))
   for direction in directions:
    direction/=np.linalg.norm(direction);values=[]
    for step in [2e-3,1e-3]:
     side=[model.evaluate(calc,xyz+sign*step*direction/ANGSTROM_TO_BOHR,properties=('energy',),reference_result=saved) for sign in [-1,1]]
     values.append([((right.energy_hartree-left.energy_hartree)/(2*step),(right.energy_hartree-2*center.energy_hartree+left.energy_hartree)/step**2) for center,left,right in zip(result.states,side[0].states,side[1].states)])
    richardson=(4*np.array(values[1])-np.array(values[0]))/3
    for i,state in enumerate(result.states):
     gradient=float(np.sum(state.gradient_hartree_per_bohr*direction));hessian=float(direction.ravel()@state.hessian_hartree_per_bohr2@direction.ravel())
     check=dict(state_id=state.state_id,gradient_error_hartree_per_bohr=abs(gradient-richardson[i,0]),curvature_error_hartree_per_bohr2=abs(hessian-richardson[i,1]),Hessian_symmetry_error=float(np.max(abs(state.hessian_hartree_per_bohr2-state.hessian_hartree_per_bohr2.T))))
     assert check['gradient_error_hartree_per_bohr']<1e-7 and check['curvature_error_hartree_per_bohr2']<2e-5,check
     checks.append(check)
  report=dict(host=socket.getfqdn(),model=model.fingerprint,static_self_energy_subtraction=True,energy_max_error_ev=energy_error,spin_squared=spin,restart_error_hartree=restart,public_API_seconds=seconds,complete_analytic_Hessian=a.hessian,hessian_workers=a.hessian_workers,serial_matrix_comparison=serial_comparison,independent_directional_energy_validation=checks,DFT_imports=0,new_SCC_calls=0,TDA_solves=0,finite_differences_in_production=False,scope='Frozen anchored molecular model; no pigment nuclear derivative qualification')
  certificate_path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
