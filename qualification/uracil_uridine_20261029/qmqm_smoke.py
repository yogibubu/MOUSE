from remote_bootstrap import HOST
from matrix_link.optimizer import _prepare_resident_oniom_component
from matrix_link.scan import QMScanBackend
from matrix_link.fragment_oniom import ResidentCappedSubtractiveSession
from pathlib import Path
import numpy as np,json
x=np.array([[0.,0.,0.],[0.,0.,.85]])
spec=QMScanBackend(name='matrix_gfn2',electronic_state=1,native_excited_method='roks',deltascf_promotion=(0,1))
components=[_prepare_resident_oniom_component(spec,('H','H'),x,topology_directory=Path('qmqm_smoke')/str(i)) for i in range(3)]
session=ResidentCappedSubtractiveSession(('H','H'),x,model_atom_indices=(0,1),covalent_bonds=((0,1),),low_real_calculator=components[0],high_model_calculator=components[1],low_model_calculator=components[2])
a=session.evaluate(x,compute_hessian=True);session.accept(a.component_results)
b=session.evaluate(x+np.array([[0.,0.,0.],[0.,0.,.001]]),compute_hessian=True)
assert all(c.reference is a.component_results[i].scf_state for i,c in enumerate(components))
assert abs(b.energy_hartree-b.component_results[1].energy_hartree)<1e-12
assert np.max(abs(b.gradient_hartree_per_bohr-b.component_results[1].gradient_hartree_per_bohr))<1e-12
assert np.max(abs(b.hessian_hartree_per_bohr2-b.component_results[1].hessian_hartree_per_bohr2))<1e-12
Path('qmqm_smoke.json').write_text(json.dumps(dict(host=HOST,passed=True,scope='three real ROKS components; identical high/low algebraic cancellation; no cap-boundary molecular validation claimed',trial_overlaps=[r.execution['state_assignment_overlap'] for r in b.component_results]),indent=2))
print('QM/QM analytic E/g/H and three accepted-state histories passed')
