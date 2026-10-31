from remote_bootstrap import HOST
from benchmark import BASE,write
from matrix_link.optimizer import _prepare_resident_oniom_component
from matrix_link.scan import QMScanBackend
from pathlib import Path
import json,numpy as np
meta=json.loads((BASE/'inputs/uracil_s0.json').read_text());x=np.array(meta['coordinates_angstrom']);atoms=tuple({1:'H',6:'C',7:'N',8:'O'}[z] for z in meta['atomic_numbers'])
spec=QMScanBackend(name='matrix_gfn2',electronic_state=1,native_excited_method='roks',deltascf_promotion=(20,21),orbital_saddle_index=1)
c=_prepare_resident_oniom_component(spec,atoms,x,topology_directory=BASE/'link_gmf_smoke')
a=c.evaluate(x,compute_hessian=True);assert c.reference is None
assert a.execution['orbital_saddle_index']==a.execution['target_orbital_saddle_index']==1
assert a.hessian_hartree_per_bohr2.shape==(36,36)
c.accept(a);trial=c.evaluate(x+np.eye(len(x),3)*1e-5)
assert c.reference is a.scf_state and trial.execution['state_assignment_overlap']>.99
write(BASE/'link_gmf_smoke.json',dict(host=HOST,passed=True,energy=a.energy_hartree,trial_execution=trial.execution))
print('LINK GMF saddle index, analytic Hessian and accepted-state tracking passed')
