from remote_bootstrap import HOST,BASE
from pathlib import Path
from types import SimpleNamespace
import sys,json,numpy as np
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import plan,state_from,write
from matrix_gfn2 import GFN2Calculator,load_response_reference
from matrix_gfn2.deltascf_derivatives import load_derivative_point
from matrix_link.scan import QMScanBackend
from matrix_link.deltascf_component import evaluate_component
from pyscf import lib
lib.num_threads(2)
c=next(c for c in plan() if c['name']=='nitroxyl');target=c['targets'][0];r0,xyz=state_from(target);s=r0.reference_state
model=BASE/'results/models/nitroxyl/tensor_charge_spin.npz';ref=load_response_reference(model)
out=BASE/'results/derivatives/nitroxyl/tensor_charge_spin'
with np.load(out/'central.npz') as f:request=str(f['request_hash'])
central=load_derivative_point(out/'central.npz',request_hash=request,coordinates_angstrom=xyz)
b=QMScanBackend(name='matrix_gfn2',electronic_state=1,native_excited_method='roks',native_roks_response_path=model,
 deltascf_promotion=(target['hole'],target['particle']),orbital_saddle_index=central.orbital_saddle_index)
calc=GFN2Calculator([{1:'H',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
accepted=(central,central.to_apoc(1));component=SimpleNamespace(backend=b,calculator=SimpleNamespace(_calculator=calc),reference=accepted,anchor=accepted,
 _cross_overlap=lambda previous,current:s.overlap)
ground=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=r0.ground_energy_hartree)
p=evaluate_component(component,xyz,True,ground,{})
with np.load(out/'analytic.npz') as f:h=f['hessian']
report=dict(host=HOST,energy_error=abs(p.energy_hartree-central.energy_hartree),gradient_error=float(np.max(abs(p.gradient_hartree_per_bohr-central.gradient_hartree_per_bohr.ravel()))),
 hessian_error=float(np.max(abs(p.hessian_hartree_per_bohr2-h))),overlap=p.execution['state_assignment_overlap'],
 response_fingerprint=p.execution['no_offset_response_fingerprint'],energy_anchor=p.execution['frozen_roks_anchor_fingerprint'])
report['passed']=bool(report['energy_error']<1e-10 and report['gradient_error']<1e-9 and report['hessian_error']<1e-8 and report['overlap']>.999999 and report['energy_anchor'] is None)
write(BASE/'results/link_molecular_validation.json',report);print(report);assert report['passed']
