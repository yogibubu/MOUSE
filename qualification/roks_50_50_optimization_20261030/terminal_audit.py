"""Audit the last accepted CSFs without reoptimizing a terminated geometry."""
import sys,importlib.util,json,traceback
from pathlib import Path
BASE=Path('/home/enzo/MOUSE-roks-50-50-optimization-20261030')
spec=importlib.util.spec_from_file_location('campaign',BASE/'source/qualification/run_campaign_recovered.py');campaign=importlib.util.module_from_spec(spec);spec.loader.exec_module(campaign)
from matrix_gfn2 import load_response_reference,evaluate_composite_roks_analytic_hessian
from matrix_link.qmqm_component import CappedQMComponent
from types import SimpleNamespace
np=campaign.np
with campaign.threadpool_limits(limits=2):
 for name in sys.argv[1:]:
    root=BASE/'results'/name
    service,settings,model,backend,_=campaign.service_for(name)
    pair=service._orbital_component.reference;result=pair[0]
    xyz=result.components[0].reference_state.coordinates_bohr/campaign.ANGSTROM_TO_BOHR
    record=dict(host=campaign.HOST,molecule=name,role='last accepted; not a certified minimum',
        energy_hartree=result.energy_hartree,maximum_gradient=float(np.max(abs(result.gradient_hartree_per_bohr))),
        coordinates_angstrom=xyz.tolist(),component_csf_overlap=result.component_csf_overlap,indices=list(result.orbital_saddle_indices))
    try:
        refs=tuple(load_response_reference(p) for p in backend.native_roks_composite_response_paths)
        h=evaluate_composite_roks_analytic_hessian((service._matrix_gfn2_session._calculator,)*2,xyz,refs,reference_result=result)
        modes=campaign._exact_cartesian_normal_modes(campaign.prepare(name)['atoms'],xyz,h.hessian_hartree_per_bohr2)
        record.update(minimum_absolute_orbital_curvatures_hartree=[c.minimum_absolute_orbital_curvature_hartree for c in h.components],frequencies_cm=modes.frequencies_cm.tolist())
        np.savez_compressed(root/'terminal_accepted.npz',coordinates_angstrom=xyz,gradient=result.gradient_hartree_per_bohr,hessian=h.hessian_hartree_per_bohr2)
    except Exception as exc:record['hessian_failure']=str(exc)
    campaign.write(root/'terminal_audit.json',record);print(json.dumps(record),flush=True)
