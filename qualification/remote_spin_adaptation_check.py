from remote_bootstrap import BASE,HOST
import json
import numpy as np
from matrix_gfn2 import GFN2Calculator,spin_adapt_gfn2_singlet,evaluate_gfn2_deltascf_analytic_hessian
from matrix_gfn2.deltascf_derivatives import load_derivative_point
records=[]
for name,atoms in [('h2',['H','H']),('formaldehyde',['C','O','H','H'])]:
    path=next((BASE/'results'/name/'deltascf').glob('*/central.npz'))
    request=json.loads((path.parent/'request.json').read_text());xyz=np.asarray(request['xyz'])
    bs=load_derivative_point(path,request_hash=path.parent.name,coordinates_angstrom=xyz)
    calc=GFN2Calculator(atoms,temperature_kelvin=0.)
    pure=spin_adapt_gfn2_singlet(calc,bs)
    rp=next((BASE/'results'/name/'roks').glob('*/central.npz'))
    reference=load_derivative_point(rp,request_hash=rp.parent.name,coordinates_angstrom=xyz)
    print(name,'adapted energy',pure.energy_hartree,'independent energy',reference.energy_hartree,flush=True)
    assert pure.spin_squared==0 and abs(pure.energy_hartree-reference.energy_hartree)<1e-9
    analytic=evaluate_gfn2_deltascf_analytic_hessian(calc,xyz,*pure.occupations,mode='roks',reference_result=pure)
    h=analytic.hessian_hartree_per_bohr2;g=pure.gradient_hartree_per_bohr
    r=xyz*1.8897261254578281
    rotation_error=0.
    for axis in np.eye(3):
        tangent=np.cross(axis,r)
        expected=np.cross(axis,g)
        rotation_error=max(rotation_error,float(np.max(abs((h @ tangent.ravel()).reshape(g.shape)-expected))))
    assert rotation_error<1e-8
    record=dict(host=HOST,molecule=name,bs_spin_squared=bs.spin_squared,pure_spin_squared=pure.spin_squared,
        roks_energy_from_bs_hartree=pure.energy_hartree,
        roks_energy_difference_from_independent_seed_hartree=pure.energy_hartree-reference.energy_hartree,
        rotational_covariance_residual_hartree_per_bohr=rotation_error)
    records.append(record);print(json.dumps(record),flush=True)
(BASE/'results'/'spin_adaptation.json').write_text(json.dumps(records,indent=2)+'\n')
