"""Coupled-stationarity and analytic nuclear response of shared-active ROKS."""
import os,socket,sys,json
from pathlib import Path
import numpy as np
import pytest

@pytest.fixture
def problem():
    assert socket.getfqdn()=='oracle.sns.it'
    from matrix_gfn2 import GFN2Calculator,load_response_reference
    from matrix_gfn2.deltascf_derivatives import load_derivative_point
    root=Path('/home/enzo/MOUSE-roks-local-20261030/results/formamide')
    data=json.loads((root/'prepared.json').read_text());accepted=root/'optimization/roks_50_50_accepted';manifest=json.loads((accepted/'accepted.json').read_text())
    parents=[]
    for row in manifest['reference']:
        path=accepted/row['path']
        with np.load(path) as d:xyz=d['coordinates_angstrom'].copy();request=str(d['request_hash'])
        parents.append(load_derivative_point(path,request_hash=request,coordinates_angstrom=xyz))
    refs=[load_response_reference(path) for path in data['response_paths']]
    calc=GFN2Calculator(data['atoms'],temperature_kelvin=0.)
    calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('SCC invoked'))
    return calc,xyz,refs,parents


@pytest.mark.parametrize('shared',['active_pair','hole'])
def test_coupled_nuclear_hessian_and_nonstationary_parent_residuals(problem,tmp_path,shared):
    from matrix_gfn2.shared_pair_roks import evaluate_shared_pair_roks,evaluate_shared_pair_roks_analytic_hessian,_contexts,_SharedPairObjective
    calc,xyz,refs,parents=problem
    result=evaluate_shared_pair_roks(calc,xyz,refs,initial_components=parents,gradient=True,
        shared_orbitals=shared,freeze_and_release=shared=='active_pair')
    assert result.converged and result.orbital_residual<1e-9
    assert max(result.component_unconstrained_residuals)>1e-3
    assert all(not r.converged and r.orbital_saddle_index is None for r in result.components)
    assert result.spin_squared==0.
    from matrix_gfn2.shared_pair_roks import save_shared_pair_roks,load_shared_pair_roks
    save_shared_pair_roks(tmp_path/'joint.npz',result,request_hash='molecular_coupled_response')
    restored=load_shared_pair_roks(tmp_path/'joint.npz',request_hash='molecular_coupled_response')
    assert abs(restored.energy_hartree-result.energy_hartree)<1e-12
    assert np.max(abs(restored.gradient_hartree_per_bohr-result.gradient_hartree_per_bohr))==0.
    assert all(not parent.converged for parent in restored.components)
    with pytest.raises(ValueError,match='another request'):
        load_shared_pair_roks(tmp_path/'joint.npz',request_hash='independent_parent_model')
    result=restored
    opens=result.shared_orbital_indices
    assert np.max(abs(result.coefficients[0][:,opens]-result.coefficients[1][:,opens]))<1e-10
    if shared=='hole':
        assert len(opens)==1 and result.component_active_pair_overlap<.999
        with pytest.raises(ValueError,match='different shared'):
            evaluate_shared_pair_roks(calc,xyz,refs,reference_result=result,shared_orbitals='active_pair')
    from matrix_gfn2 import reference_roks_to_native_ground,ground_referenced_roks_analytic_hessian
    total=reference_roks_to_native_ground(calc,xyz,refs,excited=result)
    h=ground_referenced_roks_analytic_hessian(calc,xyz,refs,reference_result=total)
    contexts=_contexts(calc,xyz,refs,result.components);obj=_SharedPairObjective(contexts,result.occupations,result.shared_orbital_indices)
    _,_,electronic=obj.evaluate(result.coefficients)
    assert np.max(abs(h.excited.orbital_hessian_hartree-electronic))<1e-9
    assert h.excited.maximum_response_residual<1e-10
    direction=np.sin(np.arange(xyz.size)+.3).reshape(xyz.shape);direction-=direction.mean(axis=0);direction/=np.linalg.norm(direction)
    from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
    errors=[]
    for step in [2e-4,1e-4]:
        points=[]
        for sign in [-1,1]:
            points.append(evaluate_shared_pair_roks(calc,xyz+sign*step/ANGSTROM_TO_BOHR*direction,refs,
                reference_result=result,gradient=True))
        points=[reference_roks_to_native_ground(calc,xyz+sign*step/ANGSTROM_TO_BOHR*direction,refs,excited=excited,reference_result=total) for sign,excited in zip([-1,1],points)]
        a,b=points
        assert abs((b.energy_hartree-a.energy_hartree)/(2*step)-np.sum(total.gradient_hartree_per_bohr*direction))<1e-7
        errors.append(np.max(abs((b.gradient_hartree_per_bohr-a.gradient_hartree_per_bohr).ravel()/(2*step)-h.hessian_hartree_per_bohr2@direction.ravel())))
    assert errors[1]<1e-7 and errors[1]<.4*errors[0]


@pytest.mark.parametrize('coupling',['shared_active_pair','shared_active_hole','relative_active_pair'])
def test_automatic_shared_pair_link_and_atomic_restart(problem,tmp_path,coupling):
    from matrix_link.scan import QMScanBackend
    from matrix_link.qmqm_component import CappedQMComponent
    from matrix_gfn2 import combine_roks_results
    from types import SimpleNamespace
    calc,xyz,refs,parents=problem
    data=json.loads(Path('/home/enzo/MOUSE-roks-local-20261030/results/formamide/prepared.json').read_text())
    backend=QMScanBackend(name='matrix_gfn2',electronic_state=1,native_excited_method='roks_50_50',
        deltascf_promotion=tuple(data['promotion']),native_roks_composite_response_paths=tuple(data['response_paths']),
        native_roks_composite_saddle_indices=tuple(data['saddle_indices']),native_roks_composite_reference_policy='geometry_only',
        native_roks_composite_component_tracking='active_pair',native_roks_composite_following='character_newton_index',
        native_roks_composite_ground_surface='native_plus_gaps',native_roks_composite_orbital_coupling=coupling)
    if coupling=='relative_active_pair':
        from dataclasses import replace
        root=Path('/home/enzo/MOUSE-roks-local-20261030/results/formamide')
        backend=replace(backend,native_roks_composite_seed_paths=tuple(root/f'anchor_{i}.npz' for i in range(2)))
        with pytest.raises(ValueError,match='seed checksums'):
            replace(backend,native_roks_composite_seed_sha256=('wrong','wrong'))
    component=CappedQMComponent(backend,SimpleNamespace(_calculator=calc),tmp_path)
    if coupling!='relative_active_pair':
        independent=combine_roks_results(*parents,component_tracking='active_pair')
        component.reference=(independent,tuple(r.to_apoc(1) for r in parents));component.anchor=component.reference
    else:
        assert component.reference is not None and component.anchor is component.reference
    point=component.evaluate(xyz,compute_hessian=True)
    assert point.execution['orbital_residual']<1e-9
    assert max(point.execution['component_unconstrained_residuals'])>1e-3
    assert point.execution['hessian_source']==dict(shared_active_pair='analytic_shared_pair_coupled_response_plus_ground_replacement',shared_active_hole='analytic_shared_hole_coupled_response_plus_ground_replacement',relative_active_pair='analytic_relative_pair_kkt_response_plus_ground_replacement')[coupling]
    component.accept(point)
    restored=CappedQMComponent(backend,SimpleNamespace(_calculator=calc),tmp_path)
    repeated=restored.evaluate(xyz)
    assert abs(repeated.energy_hartree-point.energy_hartree)<1e-11
    assert np.max(abs(repeated.gradient_hartree_per_bohr-point.gradient_hartree_per_bohr))<1e-10
    assert all(not p.converged for p in restored.reference[0].components)
    assert restored.reference[0].excited.converged
    from dataclasses import replace
    with pytest.raises(ValueError,match='contract'):
        CappedQMComponent(replace(backend,native_roks_composite_orbital_coupling=None),SimpleNamespace(_calculator=calc),tmp_path)
