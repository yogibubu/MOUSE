"""Actual molecular LINK/APOC anchored ROKS dispatch, Oracle only."""
from remote_bootstrap import HOST,BASE
from pathlib import Path
from types import SimpleNamespace
import json,numpy as np
from spin_kernel_probe import write,EV,state_from
from matrix_gfn2 import GFN2Calculator,load_frozen_roks_anchor,load_anchored_roks_point
from matrix_link.scan import QMScanBackend
from matrix_link.deltascf_component import evaluate_component
from pyscf import lib
lib.num_threads(2)
out=BASE/'results/relaxed_anchors/nitroxyl/n_pi'
a=load_frozen_roks_anchor(out/'model.npz');v=json.loads((out/'validation.json').read_text())
case=next(c for c in json.loads((BASE/'results/full_reference_inputs/frozen_relaxation/plan.json').read_text())['cases'] if c['name']=='nitroxyl')
r0,xyz=state_from(case['targets'][0]);s=r0.reference_state
central=load_anchored_roks_point(out/'central.npz',a,request_hash=v['request_hash'],coordinates_angstrom=xyz)
b=json.loads((out/'backend.json').read_text());backend=QMScanBackend(**b)
calc=GFN2Calculator([{1:'H',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
accepted=(central,central.to_apoc(backend.electronic_state))
component=SimpleNamespace(backend=backend,calculator=SimpleNamespace(_calculator=calc),reference=accepted,anchor=accepted,
 _cross_overlap=lambda old,new:s.overlap)
ground=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=r0.ground_energy_hartree)
point=evaluate_component(component,xyz,True,ground,{})
with np.load(out/'analytic.npz') as f:err=float(np.max(abs(point.hessian_hartree_per_bohr2-f['hessian'])))
report=dict(host=HOST,molecule='nitroxyl',energy_error=abs(point.energy_hartree-central.energy_hartree),
 gradient_error=float(np.max(abs(point.gradient_hartree_per_bohr.reshape(xyz.shape)-central.gradient_hartree_per_bohr))),
 hessian_error=err,reference_overlap=point.execution['state_assignment_overlap'],anchor_overlap=point.execution['state_anchor_overlap'],
 hessian_source=point.execution['hessian_source'],model_fingerprint=point.execution['frozen_roks_anchor_fingerprint'])
report['passed']=bool(report['energy_error']<1e-10 and report['gradient_error']<1e-9 and err<1e-8 and report['reference_overlap']>.999999)
write(BASE/'results/link_molecular_validation.json',report);print(report);assert report['passed']
