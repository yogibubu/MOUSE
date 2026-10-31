from remote_bootstrap import BASE,HOST
import json
import numpy as np
from matrix_gfn2 import GFN2Calculator,evaluate_gfn2_deltascf_hessian,evaluate_gfn2_deltascf_analytic_hessian
from matrix_gfn2.deltascf_derivatives import load_derivative_point
records=[]
for name,atoms in [('h2',['H','H']),('formaldehyde',['C','O','H','H'])]:
 for mode in ('deltascf','roks'):
    # Recover the already completed central molecular point; no new central SCF.
    path=next((BASE/'results'/name/mode).glob('*/central.npz'))
    request=json.loads((path.parent/'request.json').read_text())
    xyz=np.asarray(request['xyz']);alpha=np.asarray(request['alpha']);beta=np.asarray(request['beta'])
    central=load_derivative_point(path,request_hash=path.parent.name,coordinates_angstrom=xyz)
    calc=GFN2Calculator(atoms,temperature_kelvin=0.)
    analytic=evaluate_gfn2_deltascf_analytic_hessian(calc,xyz,alpha,beta,mode=mode,reference_result=central)
    opts=request['solver']
    errors=[]
    for step in (.0005,.00025):
        numeric=evaluate_gfn2_deltascf_hessian(calc,xyz,alpha,beta,mode=mode,step_bohr=step,
            checkpoint_dir=BASE/'results'/'refinement'/name/mode,reference_result=central,**opts)
        error=float(np.max(np.abs(analytic.hessian_hartree_per_bohr2-numeric.hessian_hartree_per_bohr2)))
        translation=np.tile(np.eye(3),(len(atoms),1))
        record=dict(host=HOST,molecule=name,mode=mode,step_bohr=step,hessian_max_error=error,
            response_residual=analytic.maximum_response_residual,
            translation_residual=float(np.max(np.abs(analytic.hessian_hartree_per_bohr2 @ translation))),
            new_points=numeric.evaluations,reused_points=numeric.reused_points)
        records.append(record);errors.append(error);print(json.dumps(record),flush=True)
    repeated=evaluate_gfn2_deltascf_hessian(calc,xyz,alpha,beta,mode=mode,step_bohr=.00025,
        checkpoint_dir=BASE/'results'/'refinement'/name/mode,reference_result=central,**opts)
    assert repeated.evaluations==0 and repeated.reused_points==1+6*len(atoms)
    assert errors[-1] < 1e-6
    assert errors[-1] < errors[0]/2
(BASE/'results'/'derivative_refinement.json').write_text(json.dumps(records,indent=2)+'\n')
