"""Four explicitly budgeted, checkpointed accepted steps to the pi-pi geometry."""
from remote_bootstrap import HOST
from benchmark import BASE,write,EV
import json,numpy as np,time
from matrix_gfn2 import GFN2Calculator,evaluate_gfn2_roks_singlet
from matrix_gfn2.deltascf_derivatives import load_derivative_point,save_derivative_point
name='uridine';target='pi_pi';start=BASE/'results/uridine_s0';end=BASE/'results/uridine_pi_pi'
d0=json.loads((start/'summary.json').read_text());d1=json.loads((end/'summary.json').read_text());x0=np.array(d0['input']['coordinates_angstrom']);x1=np.array(d1['input']['coordinates_angstrom'])
r=load_derivative_point(start/'qualified/roks_pi_pi.npz',request_hash='uridine_s0',coordinates_angstrom=x0)
folder=end/'continuation_pi_pi';folder.mkdir(exist_ok=True)
records=[]
for index in range(1,5):
 x=x0+(x1-x0)*index/4;path=folder/f'step_{index}.npz';key=f'uridine_pi_continuation_{index}'
 t=time.perf_counter()
 try:
  if path.exists():rr=load_derivative_point(path,request_hash=key,coordinates_angstrom=x)
  else:
   calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[z] for z in d0['input']['atomic_numbers']],temperature_kelvin=0.)
   rr=evaluate_gfn2_roks_singlet(calc,x,*r.occupations,reference_result=r,gradient=True,minimum_state_overlap=.70,orbital_tolerance=1e-9,maximum_evaluations=80,rechart_interval=6)
   save_derivative_point(path,rr,request_hash=key,coordinates_angstrom=x)
  records.append(dict(step=index,seconds=time.perf_counter()-t,gap_ev=rr.excitation_energy_hartree*EV,residual=rr.orbital_residual));r=rr
  if index==4:save_derivative_point(end/'qualified/roks_pi_pi.npz',r,request_hash='uridine_pi_pi',coordinates_angstrom=x1)
 except Exception as error:
  records.append(dict(step=index,seconds=time.perf_counter()-t,error=str(error)));write(folder/'status.json',records);break
 write(folder/'status.json',records)
