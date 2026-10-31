from remote_bootstrap import BASE,HOST
import json,time
import numpy as np
from matrix_gfn2 import GFN2Calculator, evaluate_gfn2_deltascf_analytic_hessian, evaluate_gfn2_deltascf_hessian
from matrix_gfn2.response_reference import stationary_response_reference
for name,atoms,xyz in [('h2',['H','H'],np.array([[0.,0.,0.],[0.,0.,.85]])),
 ('formaldehyde',['C','O','H','H'],np.array([[0.,0.,0.],[1.21,0.,0.],[-.58,.94,0.],[-.58,-.94,0.]]))]:
 for mode in ('deltascf','roks'):
    calc=GFN2Calculator(atoms,temperature_kelvin=0.)
    ground,_=stationary_response_reference(calc,xyz)
    state=ground.orbital_state
    alpha=state.alpha_occupations.copy();beta=state.beta_occupations.copy()
    hole=np.flatnonzero(alpha>.5)[-1];particle=np.flatnonzero(alpha<.5)[0]
    alpha[hole]=0;alpha[particle]=1
    options = (dict(maximum_iterations=1000,density_tolerance=1e-10,orbital_tolerance=1e-10,energy_tolerance=1e-12)
               if mode=='deltascf' else dict(orbital_tolerance=1e-10))
    start=time.monotonic()
    analytic=evaluate_gfn2_deltascf_analytic_hessian(calc,xyz,alpha,beta,mode=mode,ground_state=ground,**options)
    print(name,mode,'analytic complete',time.monotonic()-start,'sym',analytic.raw_symmetry_error_hartree_per_bohr2,
       'res',analytic.maximum_response_residual,'curvature',analytic.minimum_absolute_orbital_curvature_hartree,flush=True)
    numeric=evaluate_gfn2_deltascf_hessian(calc,xyz,alpha,beta,mode=mode,step_bohr=.001,
        checkpoint_dir=BASE/'results'/name/mode,reference_result=analytic.reference,**options)
    error=float(np.max(np.abs(analytic.hessian_hartree_per_bohr2-numeric.hessian_hartree_per_bohr2)))
    record=dict(host=HOST,molecule=name,mode=mode,hessian_max_error=error,
       analytic_asymmetry=analytic.raw_symmetry_error_hartree_per_bohr2,numeric_asymmetry=numeric.symmetry_error_hartree_per_bohr2,
       response_residual=analytic.maximum_response_residual,seconds=time.monotonic()-start,
       analytic_hessian=analytic.hessian_hartree_per_bohr2.tolist(),numeric_hessian=numeric.hessian_hartree_per_bohr2.tolist())
    (BASE/'results'/f'{name}_{mode}_hessian.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if 'hessian' not in k or k=='hessian_max_error'}),flush=True)
