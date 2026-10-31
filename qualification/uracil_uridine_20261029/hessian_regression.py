"""Recheck the final Hessian implementation using completed molecular checkpoints."""
from remote_bootstrap import BASE,HOST
import json
import numpy as np
from matrix_gfn2 import GFN2Calculator,evaluate_gfn2_deltascf_analytic_hessian
from matrix_gfn2.deltascf_derivatives import load_derivative_point
records=[]
for name,atoms in [('h2',['H','H']),('formaldehyde',['C','O','H','H'])]:
 for mode in ('deltascf','roks'):
    path=next((BASE/'results'/name/mode).glob('*/central.npz'))
    request=json.loads((path.parent/'request.json').read_text());xyz=np.asarray(request['xyz'])
    center=load_derivative_point(path,request_hash=path.parent.name,coordinates_angstrom=xyz)
    calc=GFN2Calculator(atoms,temperature_kelvin=0.)
    h=evaluate_gfn2_deltascf_analytic_hessian(calc,xyz,*center.occupations,mode=mode,reference_result=center)
    old=json.loads((BASE/'results'/f'{name}_{mode}_hessian.json').read_text())
    discrepancy=float(np.max(np.abs(h.hessian_hartree_per_bohr2-np.array(old['analytic_hessian']))))
    assert discrepancy<1e-10
    record=dict(host=HOST,molecule=name,mode=mode,final_implementation_hessian_difference=discrepancy,
        response_residual=h.maximum_response_residual,new_scf_points=0)
    records.append(record);print(json.dumps(record),flush=True)
(BASE/'uracil_uridine_20261029/hessian_regression.json').write_text(json.dumps(records,indent=2)+'\n')

from pathlib import Path
import time
from benchmark import BASE as campaign
records2=[]
for name,target in [('uracil_s0','n_pi'),('uracil_s0','pi_pi'),('uracil_n_pi','n_pi'),('uracil_pi_pi','pi_pi')]:
 out=campaign/'results'/name
 d=json.loads((out/'summary.json').read_text());xyz=np.array(d['input']['coordinates_angstrom'])
 r=load_derivative_point(out/f'roks_{target}.npz',request_hash=name,coordinates_angstrom=xyz)
 calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[z] for z in d['input']['atomic_numbers']],temperature_kelvin=0.)
 t=time.perf_counter();h=evaluate_gfn2_deltascf_analytic_hessian(calc,xyz,*r.occupations,mode='roks',reference_result=r)
 elapsed=time.perf_counter()-t
 old=np.load(out/f'roks_hessian_{target}.npz')['hessian']
 error=float(np.max(abs(h.hessian_hartree_per_bohr2-old)));assert error<1e-10
 records2.append(dict(name=name,target=target,seconds=elapsed,max_difference=error,response_residual=h.maximum_response_residual))
(campaign/'molecular_hessian_regression.json').write_text(json.dumps(records2,indent=2))
