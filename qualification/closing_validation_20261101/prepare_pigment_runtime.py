"""Export full frozen-XC quadrature and an RI orbital-solver preconditioner.

The portable vertical runtime loads NumPy arrays only. All functionals,
quadrature, source checks and native integral jobs run on verified MAC-STUDIO.
"""
from pigment_compact_reference import HERE,Compact,load_state
from pigment_reference_embedding import write,sha
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.nonlocal_response import FrozenNonlocalResponseReference
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from pathlib import Path
from threadpoolctl import threadpool_limits
import numpy as np,json,time,os,argparse,shutil,socket

def export(name):
 out=HERE/'pigment_models'/name;out.mkdir(parents=True,exist_ok=True);grid=out/'quadrature';grid.mkdir(exist_ok=True)
 if not (grid/'manifest.json').exists():
  case=Compact(name);s=case.state
  with np.load(case.out/'anchor.npz') as z:p=z['density'].copy()
  rows=[]
  for k,path in enumerate(case.grid):
   target=grid/f'chunk_{k:04d}.npz'
   if not target.exists():
    with np.load(path) as z:phi=z['phi'];weight=z['weight'];rho=.5*(z['core']+case.rho(phi,p))
    f=case.mf._numint.eval_xc_eff('CAM-B3LYP',np.array([rho,rho]),deriv=2,xctype='GGA')[2];charge=.25*(f[0,:,0]+f[0,:,1]+f[1,:,0]+f[1,:,1]);spin=.25*(f[0,:,0]-f[0,:,1]-f[1,:,0]+f[1,:,1]);temporary=target.with_suffix('.tmp.npz');np.savez(temporary,phi=phi,weight=weight,charge=charge,spin=spin);temporary.replace(target)
   rows.append(dict(file=target.name,sha256=sha(target)))
  write(grid/'manifest.json',dict(coordinates_bohr=s.coordinates_bohr.tolist(),chunks=rows,xc='CAM-B3LYP',exchange_response_scale=.9,density_center='frozen embedded CAM anchor',role='Exact quadrature correction to RI auxiliary linearization; no energy offset'))
 s=load_state(name);shutil.copy2(HERE/'pigment_preflight'/name/'stationary_native.npz',out/'state.npz');prepared=HERE/'pigment_full_response'/name/'ri_xc_chart'
 while not (prepared/'certificate.json').exists():time.sleep(20)
 if not (out/'response.npz').exists():
  calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O',12:'Mg'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.);ctx=prepare_gfn2_deltascf(calc,s.coordinates_bohr/ANGSTROM_TO_BOHR,orbital_reference=s,allow_large=True)
  with np.load(prepared/'field.npz') as z:linear=z['linear'].copy();q0=z['q0'].copy()
  data=dict(operators=np.load(prepared/'operators.npy',mmap_mode='r'),charge=np.load(prepared/'charge.npy',mmap_mode='r'),spin=np.load(prepared/'spin.npy',mmap_mode='r'),q0=q0,linear=linear)
  reference=FrozenNonlocalResponseReference.from_vertical_model(s,ctx.kernel.partition,data,sha(prepared/'certificate.json')+':RI_XC_linearization_only')
  arrays={field:getattr(reference,field) for field in reference.__dataclass_fields__};arrays.update(schema='mouse.nonlocal_response.v1',fingerprint=reference.fingerprint);temporary=out/'response.tmp.npz';np.savez(temporary,**arrays);temporary.replace(out/'response.npz')
 for label,index in [('charge',0),('spin',1)]:
  path=out/f'xc_{label}.npy'
  if not path.exists():os.link(prepared/'ri_xc'/f'kernel_multipole_{index}.npy',path)
 write(out/'manifest.json',dict(schema='mouse.pigment.vertical_quadrature.v1',molecule=name,host=socket.getfqdn(),sha256={f:sha(out/f) for f in ['state.npz','response.npz','xc_charge.npy','xc_spin.npy']},quadrature_manifest_sha256=sha(grid/'manifest.json'),weights=[.5,.5],lambda_exchange=.9,bath_efficiency_eta=.6,energy_offset_ev=0,exact_frozen_XC_residual=True,RI_XC_role='Newton linearization only; exact frozen-quadrature energy and residual correction',scope='Vertical ROKS; nuclear derivative adapter not yet qualified on pigments'))
 print(name,'portable exact-XC vertical runtime ready',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--wait-for-anchor',action='store_true');args=p.parse_args()
 while not (HERE/'pigment_compact_reference'/args.name/'certificate.json').exists():
  if not args.wait_for_anchor:raise RuntimeError('Compact anchor incomplete')
  time.sleep(20)
 with threadpool_limits(8):export(args.name)
