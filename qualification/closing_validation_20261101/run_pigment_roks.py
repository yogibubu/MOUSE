"""DFT-free vertical spin-pure ROKS on the prepared full-XC pigment surface."""
from pathlib import Path
import socket,subprocess,runpy,sys,importlib.abc,argparse,time,json,os,hashlib
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
assert subprocess.check_output(['scutil','--get','ComputerName'],text=True).strip()=='Mac Studio di Vincenzo'
class NoDFT(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname=='pyscf' or fullname.startswith('pyscf.'):raise AssertionError('DFT import forbidden in prepared OO runtime')
sys.meta_path.insert(0,NoDFT());BASE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101');HERE=BASE/'oracle/MOUSE-closing-validation-20261101';runpy.run_path(str(BASE/'source/qualification/remote_bootstrap.py'))
import numpy as np
from dataclasses import fields
from matrix_gfn2.electronic_state import GFN2OrbitalState,GFN2OrbitalBasis
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.spin_reference import load_response_reference
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from matrix_gfn2._quadrature_xc import FrozenQuadratureXC
from matrix_gfn2._roks_matrix_free import solve
from threadpoolctl import threadpool_limits

def write(path,data):
 p=path.with_suffix(f'.tmp.{os.getpid()}.json');p.write_text(json.dumps(data,indent=2)+'\n');p.replace(path)

def load(name):
 model=HERE/'pigment_models'/name
 with np.load(model/'state.npz') as z:
  basis=GFN2OrbitalBasis(**{f.name:z['basis_'+f.name].copy() for f in fields(GFN2OrbitalBasis)});kw={f.name:z[f.name].copy() for f in fields(GFN2OrbitalState) if f.name!='basis'}
 kw['model_id']=str(kw['model_id']);kw['ao_shell']=tuple(kw['ao_shell'].tolist())
 for field in ['electron_count','charge','multiplicity']:kw[field]=int(kw[field])
 kw['total_energy_hartree']=float(kw['total_energy_hartree']);s=GFN2OrbitalState(**kw,basis=basis)
 reference=load_response_reference(model/'response.npz');calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O',12:'Mg'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.);calc.evaluate=lambda *x,**k:(_ for _ in ()).throw(AssertionError('New SCC forbidden'))
 base=prepare_gfn2_deltascf(calc,s.coordinates_bohr/ANGSTROM_TO_BOHR,orbital_reference=s,spin_reference=reference,allow_large=True);context=FrozenQuadratureXC(base,model/'quadrature/manifest.json',model/'xc_charge.npy',model/'xc_spin.npy',reference)
 return model,s,reference,calc,base,context

def run(name,requested_state=None):
 model,s,reference,calc,base,context=load(name);out=HERE/'pigment_roks'/name;out.mkdir(parents=True,exist_ok=True);c=s.coefficients;no=s.electron_count//2;n=len(c);p=c[:,:no]@c[:,:no].T;fg,eg=context.fock_and_energy(np.array([p,p]));ground_residual=float(np.max(abs((c.T@fg[0]@c)[:no,no:])));assert ground_residual<1e-8
 promotions=[(no-1,no)] if name=='Beta_carotene' else [(no-1,no),(no-2,no),(no-1,no+1),(no-2,no+1)];states=[];start=time.monotonic()
 for ordinal,(hole,particle) in enumerate(promotions):
  if requested_state is not None and requested_state!=ordinal+1:continue
  folder=out/f'state_{ordinal+1}';folder.mkdir(exist_ok=True);checkpoint=folder/'checkpoint.npz';certificate=folder/'certificate.json'
  if certificate.exists():states.append(json.loads(certificate.read_text()));continue
  order=[hole]+[i for i in range(no) if i!=hole]+[particle]+[i for i in range(no,n) if i!=particle];seed=c[:,order];mu=0.
  if checkpoint.exists():
   with np.load(checkpoint) as z:assert str(z['model_sha256'])==hashlib.sha256((model/'manifest.json').read_bytes()).hexdigest();seed=z['coefficients'].copy();mu=float(z['multiplier'])
  transition=np.zeros((len(promotions),no,n-no))
  for k,(hh,pp) in enumerate(promotions):transition[k,hh,pp-no]=1.
  history=[]
  def callback(coefficient,multiplier,row):
   temporary=checkpoint.with_suffix('.tmp.npz');np.savez_compressed(temporary,coefficients=coefficient,multiplier=multiplier,model_sha256=hashlib.sha256((model/'manifest.json').read_bytes()).hexdigest());temporary.replace(checkpoint);history.append(row);write(folder/'progress.json',dict(molecule=name,state=ordinal+1,hole=hole,particle=particle,row=row));print(name,ordinal+1,row,flush=True)
  state,mu,report=solve(context,seed,c,transition,mu,tolerance=1e-10,maximum_iterations=100,checkpoint=callback,linearization_context=base,local_state_following=True,krylov='minres',exact_refinement=True,residual_trust_region=True)
  result=dict(molecule=name,ordinal=ordinal+1,hole=hole,particle=particle,excitation_ev=(report['energy_hartree']-eg)*27.211386245988,ground_energy_hartree=eg,report=report,spin_squared=0.,spin_definition='coherent shared-orbital two-determinant singlet; ROKS multiplet-sum energy',DFT_imports=0,new_SCC_calls=0,TDA_solves=0,full_XC_quadrature_energy=True,RI_role='orbital KKT linearization only',model_sha256=hashlib.sha256((model/'manifest.json').read_bytes()).hexdigest());write(certificate,result);states.append(result)
 certificates=[out/f'state_{i+1}'/'certificate.json' for i in range(len(promotions))]
 if all(path.exists() for path in certificates):
  states=[json.loads(path.read_text()) for path in certificates];write(out/'result.json',dict(molecule=name,host=socket.getfqdn(),states=states,current_process_wall_seconds=time.monotonic()-start,timing_scope='This worker only; target workers may run concurrently; excludes reference preparation',ground_residual=ground_residual,weights=[.5,.5],lambda_exchange=.9,offset_ev=0,scope='Spin-pure ground-orthogonal ROKS targets before screened CID; not yet a final correlated pigment prediction'));print(name,'all ROKS targets complete',flush=True)
 else:print(name,'requested ROKS target complete; other target workers pending',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--wait-for-model',action='store_true');p.add_argument('--state',type=int,choices=[1,2,3,4]);a=p.parse_args()
 while not (HERE/'pigment_models'/a.name/'manifest.json').exists():
  if not a.wait_for_model:raise RuntimeError('Prepared model incomplete')
  time.sleep(20)
 with threadpool_limits(8):run(a.name,a.state)
