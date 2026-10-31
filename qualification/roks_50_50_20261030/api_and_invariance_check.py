"""Check public composite API and Euclidean Hessian identities on saved states."""
import sys,socket
from pathlib import Path
assert socket.getfqdn()=='oracle.sns.it'
BASE=Path('/home/enzo/MOUSE-roks-50-50-20261030');sys.path.insert(0,'/home/enzo/MOUSE-roks-independent-20261030')
import remote_bootstrap
for p in (BASE/'source/packages').glob('*/src'):sys.path.insert(0,str(p))
import numpy as np,json
from threadpoolctl import threadpool_limits
from matrix_gfn2 import (GFN2Calculator,load_response_reference,combine_roks_results,
    evaluate_composite_roks,evaluate_composite_roks_analytic_hessian)
from matrix_gfn2.deltascf_derivatives import load_derivative_point
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR

def run(name):
    folder=BASE/'results'/name;models=[];states=[];calculators=[];ward=[]
    for profile in ['full_response','qualified_spectral_response']:
        path=folder/profile/'central.npz'
        with np.load(path) as d:
            req=str(d['request_hash']);xyz=d['coordinates_angstrom'].copy()
        r=load_derivative_point(path,request_hash=req,coordinates_angstrom=xyz)
        states.append(r);models.append(load_response_reference(folder/profile/'response.npz'))
        calculators.append(GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in r.reference_state.atomic_numbers],temperature_kelvin=0.))
        with np.load(folder/profile/'analytic.npz') as d:h=d['hessian']
        g=r.gradient_hartree_per_bohr;x=xyz*ANGSTROM_TO_BOHR;x-=x.mean(axis=0)
        trans=max(float(np.max(abs(h@np.tile(axis,(len(x),1)).ravel()))) for axis in np.eye(3))
        rot=max(float(np.max(abs(h@np.cross(axis,x).ravel()-np.cross(axis,g).ravel()))) for axis in np.eye(3))
        torque=float(np.max(abs(np.cross(x,g).sum(axis=0))))
        ward.append(dict(profile=profile,hessian_translation_error=trans,hessian_rotation_identity_error=rot,torque_error=torque))
    old=combine_roks_results(*states)
    options=[dict(orbital_solver='gmf',saddle_index=r.orbital_saddle_index,orbital_tolerance=1e-11,maximum_iterations=100) for r in states]
    current=evaluate_composite_roks(calculators,xyz,*states[0].occupations,models,reference_result=old,gradient=True,component_options=options)
    ah=evaluate_composite_roks_analytic_hessian(calculators,xyz,models,reference_result=current)
    with np.load(folder/'composite.npz') as d:
        checks=dict(api_energy_error=abs(current.energy_hartree-float(d['energy_hartree'])),
            api_gradient_error=float(np.max(abs(current.gradient_hartree_per_bohr-d['gradient']))),
            api_hessian_error=float(np.max(abs(ah.hessian_hartree_per_bohr2-d['hessian']))))
    record=dict(host=socket.getfqdn(),molecule=name,ward_identities=ward,api=checks)
    record['passed']=bool(max(checks.values())<1e-8 and all(max(v for k,v in row.items() if k!='profile')<1e-8 for row in ward))
    (folder/'api_and_invariance.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True);assert record['passed']
with threadpool_limits(limits=2):
    for name in sys.argv[1:]:run(name)
