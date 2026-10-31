from remote_bootstrap import HOST
from types import SimpleNamespace
from pathlib import Path
import json,numpy as np
from matrix_gfn2 import GFN2Calculator
from matrix_link.scan import QMScanBackend
from matrix_link.optimizer import _prepare_resident_oniom_component
b=QMScanBackend(name='matrix_gfn2',electronic_state=1,native_excited_method='roks',deltascf_promotion=(0,1))
c=_prepare_resident_oniom_component(b,('H','H'),np.array([[0.,0.,0.],[0.,0.,.85]]),topology_directory=Path('link_smoke'))
assert c.calculator._calculator.temperature_kelvin==0.
x=np.array([[0.,0.,0.],[0.,0.,.85]])
a=c.evaluate(x,compute_hessian=True);assert c.reference is None;c.accept(a)
b=c.evaluate(x+np.array([[0.,0.,0.],[0.,0.,.001]]),compute_hessian=True)
assert c.reference is a.scf_state and b.execution['state_assignment_overlap']>.99
assert b.execution['spin_pure_singlet'] and b.execution['hessian_source']=='analytic'
Path('link_smoke.json').write_text(json.dumps(dict(host=HOST,initial_energy=a.energy_hartree,trial_energy=b.energy_hartree,execution=b.execution,hessian_shape=list(b.hessian_hartree_per_bohr2.shape)),indent=2,default=str))
print('LINK ROKS analytic derivative and accepted-state tracking passed')
