"""Remote all-column independent Hessian verification, with column checkpoints."""
import socket,sys,os,json,time
from pathlib import Path
BASE=Path('/home/enzo/MOUSE-roks-relative-pair-20261030');assert socket.getfqdn()=='oracle.sns.it';os.environ['MOUSE_VALIDATION_ROOT']=str(BASE)
sys.path.insert(0,str(BASE/'source/qualification'));import remote_bootstrap
import numpy as np
from threadpoolctl import threadpool_limits
from matrix_gfn2 import (GFN2Calculator,load_response_reference,load_relative_pair_roks,
    evaluate_relative_pair_roks,reference_roks_to_native_ground)
from matrix_gfn2.ground_referenced_roks import GroundReferencedROKSResult
from matrix_gfn2.restricted_ground import RestrictedGroundResult
from matrix_gfn2.deltascf_derivatives import load_derivative_point
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR

def run(name):
 root=BASE/'molecular'/name;out=root/'full_hessian_columns';out.mkdir(exist_ok=True)
 if (out/'result.json').exists():return
 data=json.loads((Path('/home/enzo/MOUSE-roks-local-20261030/results')/name/'prepared.json').read_text());refs=[load_response_reference(path) for path in data['response_paths']]
 manifest=json.loads((root/'accepted.json').read_text());folder=root/manifest['generation'];joint=load_relative_pair_roks(folder/'joint.npz',request_hash=manifest['contract']);grounds=[]
 for i in range(3):
  path=folder/f'ground_{i}.npz'
  with np.load(path) as d:xyz=d['coordinates_angstrom'].copy();request=str(d['request_hash'])
  raw=load_derivative_point(path,request_hash=request,coordinates_angstrom=xyz)
  grounds.append(RestrictedGroundResult(raw,manifest['ground_curvatures'][i]))
 point=GroundReferencedROKSResult(joint,tuple(grounds))
 with np.load(root/'analytic_stationary_0.npz') as d:h=d['hessian'].copy();xyz=d['xyz'].copy()
 assert np.max(abs(xyz-joint.components[0].reference_state.coordinates_bohr/ANGSTROM_TO_BOHR))<1e-12
 calc=GFN2Calculator(data['atoms'],temperature_kelvin=0.);calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('SCC invoked'))
 numeric=np.zeros_like(h);done=np.zeros(xyz.size,dtype=bool);gradient_errors=np.zeros(xyz.size);step=1e-4;start=time.monotonic()
 cp=out/'columns.npz'
 if cp.exists():
  with np.load(cp) as d:numeric=d['numeric'].copy();done=d['done'].copy();gradient_errors=d['gradient_errors'].copy()
 for column in range(xyz.size):
  if done[column]:continue
  direction=np.zeros(xyz.size);direction[column]=1.;values=[]
  for sign in [-1,1]:
   geom=xyz+sign*step/ANGSTROM_TO_BOHR*direction.reshape(xyz.shape)
   excited=evaluate_relative_pair_roks(calc,geom,refs,reference_result=joint,gradient=True)
   values.append(reference_roks_to_native_ground(calc,geom,refs,excited=excited,reference_result=point))
  a,b=values;numeric[:,column]=(b.gradient_hartree_per_bohr-a.gradient_hartree_per_bohr).ravel()/(2*step)
  gradient_errors[column]=abs((b.energy_hartree-a.energy_hartree)/(2*step)-point.gradient_hartree_per_bohr.ravel()[column]);done[column]=True
  temp=cp.with_suffix('.tmp')
  with temp.open('wb') as stream:np.savez_compressed(stream,numeric=numeric,done=done,gradient_errors=gradient_errors)
  temp.replace(cp);print('COLUMN',name,column,float(np.max(abs(numeric[:,column]-h[:,column]))),flush=True)
 error=float(np.max(abs(numeric-h)));gradient_error=float(np.max(gradient_errors))
 row=dict(host=socket.getfqdn(),molecule=name,columns=len(done),step_bohr=step,hessian_max_error=error,
    scalar_gradient_max_error=gradient_error,numerical_symmetry_error=float(np.max(abs(numeric-numeric.T))),seconds=time.monotonic()-start,
    passed=bool(np.all(done) and error<1e-6 and gradient_error<1e-7))
 (out/'result.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=2):
  for name in ['nitroxyl','formamide']:run(name)
