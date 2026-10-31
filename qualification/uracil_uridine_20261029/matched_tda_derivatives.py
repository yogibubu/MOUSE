"""One physically matched TDA root, with per-column analytic Hessian restart."""
from remote_bootstrap import HOST
from benchmark import BASE,write
import json,time,numpy as np
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.stda import compact_xtb_tda
from matrix_gfn2.deltascf_derivatives import load_derivative_point
name='uridine_s0';target='pi_pi';out=BASE/'results'/name
comparison=json.loads((BASE/'comparison.json').read_text());root=comparison['states'][name+'_'+target]['tda_root']-1
meta=json.loads((out/'summary.json').read_text());xyz=np.array(meta['input']['coordinates_angstrom'])
ground=load_derivative_point(out/'ground.npz',request_hash=name,coordinates_angstrom=xyz);s=ground.reference_state
calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[z] for z in meta['input']['atomic_numbers']],temperature_kelvin=0.)
folder=out/'qualified'/f'matched_tda_root_{root+1}';folder.mkdir(exist_ok=True)
report=folder/'summary.json'
if not (folder/'hessian.npz').exists():
 record=json.loads(report.read_text()) if report.exists() else dict(host=HOST,root=root+1,model='compact full-virtual constant-shift',new_scf_points=0)
 t=time.perf_counter();exc=compact_xtb_tda(s,virtual_count=int(np.count_nonzero(s.alpha_occupations<.5)),roots=10,shift_model='constant');op=calc.closed_shell_stda_hessian(s,excited=exc,root=root,spectral_range_ev=10.)
 g=ground.gradient_hartree_per_bohr+op.excitation_gradient_hartree_per_bohr
 if 'gradient_seconds' not in record:record['gradient_seconds']=time.perf_counter()-t
 np.savez_compressed(folder/'gradient.npz',gradient=g);write(report,record)
 columns=[];times=[]
 for k in range(xyz.size):
  p=folder/f'column_{k:03d}.npz'
  if p.exists():
   with np.load(p) as f:column=f['column'];seconds=float(f['seconds'])
  else:
   v=np.zeros(xyz.size);v[k]=1;t=time.perf_counter();column=op.matvec(v);seconds=time.perf_counter()-t;np.savez_compressed(p,column=column,seconds=seconds)
  columns.append(column);times.append(seconds)
  record['completed_columns']=k+1;write(report,record)
 h=np.array(columns).T
 record['hessian_seconds']=sum(times);record['hessian_asymmetry']=float(np.max(abs(h-h.T)))
 np.savez_compressed(folder/'hessian.npz',hessian=h,gradient=g);write(report,record)
