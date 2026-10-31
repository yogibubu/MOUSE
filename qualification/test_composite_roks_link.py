"""Configuration, direct dispatch and immutable accepted-history boundaries."""
from types import SimpleNamespace
from dataclasses import replace,asdict
from pathlib import Path
import numpy as np
import pytest
from matrix_link.scan import QMScanBackend,_composite_restart_json_value


def backend(tmp_path):
    from matrix_gfn2 import FrozenNonlocalResponseReference,save_response_reference
    xyz=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
    state=SimpleNamespace(atomic_numbers=np.array([6,1,1]),coordinates_bohr=xyz)
    partition=SimpleNamespace(shell_atom=np.arange(4),orbital_shell=np.array([0,1,1,1,2,3]))
    operators=np.eye(6)[None];data=dict(operators=operators,charge=np.array([[.02]]),spin=np.array([[-.01]]),q0=np.zeros(1),linear=np.array([.03]))
    ref=FrozenNonlocalResponseReference.from_vertical_model(state,partition,data,'raw');a=tmp_path/'raw.npz';b=tmp_path/'spectral.npz'
    save_response_reference(a,ref);save_response_reference(b,replace(ref,pair_linear_hartree=np.array([.04]),source_model='spectral'))
    return QMScanBackend(name='matrix_gfn2',electronic_state=1,native_excited_method='roks_50_50',deltascf_promotion=(1,2),
        native_roks_composite_response_paths=(a,b),native_roks_composite_saddle_indices=(0,1))


def test_configuration_and_restart_identity(tmp_path):
    from matrix_link.optimizer import _restart_identity_value,_electronic_restart_artifact_identity
    spec=backend(tmp_path);assert QMScanBackend(**asdict(spec))==spec
    assert len(_electronic_restart_artifact_identity(spec)['native_roks_composite_response_paths'])==2
    for serializer in (_composite_restart_json_value,_restart_identity_value):
        assert serializer(spec)['native_roks_composite_saddle_indices']==[0,1]
        assert 'native_roks_composite_response_paths' not in serializer(QMScanBackend(name='matrix_gfn2'))
    for changes in [dict(native_roks_composite_response_paths=None),dict(native_roks_composite_saddle_indices=None),
        dict(native_roks_composite_saddle_indices=(-1,0)),dict(native_roks_composite_saddle_indices=(True,0)),
        dict(orbital_saddle_index=1),dict(native_excited_method='roks'),
        dict(native_roks_composite_response_paths=(spec.native_roks_composite_response_paths[0],)*2),
        dict(native_roks_composite_response_sha256=('bad','bad'))]:
        with pytest.raises(ValueError):replace(spec,**changes)


def test_changed_response_rejected_before_evaluation(tmp_path):
    from matrix_link.composite_roks_component import _models
    spec=backend(tmp_path)
    with spec.native_roks_composite_response_paths[0].open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='changed during'):_models(spec)


def test_capped_trial_acceptance_is_atomic_for_both_states(monkeypatch,tmp_path):
    from matrix_link.qmqm_component import CappedQMComponent
    from matrix_link import composite_roks_component as adapter
    c=CappedQMComponent(backend(tmp_path),SimpleNamespace(_calculator=SimpleNamespace()),tmp_path/'run')
    ids=(SimpleNamespace(state_id=1),SimpleNamespace(state_id=1));pair=(object(),ids)
    point=SimpleNamespace(scf_state=pair,execution={'selected_electronic_state':1,'gradient_electronic_state':1})
    calls=[]
    monkeypatch.setattr(adapter,'persist_accepted',lambda comp,p: calls.append((comp.reference,p)))
    c.accept(point);assert calls==[(None,pair)] and c.anchor is pair and c.reference is pair
    def failed(*a):raise OSError('simulated checkpoint write failure')
    monkeypatch.setattr(adapter,'persist_accepted',failed)
    newer=SimpleNamespace(scf_state=(object(),ids),execution=point.execution)
    with pytest.raises(OSError):c.accept(newer)
    assert c.reference is pair and c.anchor is pair
    with pytest.raises(ValueError):c.validate_accept(SimpleNamespace(scf_state=(object(),ids[:1]),execution=point.execution))


def test_direct_optimizer_dispatch_and_acceptance(monkeypatch,tmp_path):
    from matrix_link.optimizer import GeometryEvaluationService
    from matrix_link.scan import ScanPoint,PointEvaluationResult
    spec=backend(tmp_path);calls=[]
    result=PointEvaluationResult(0,0.,energy_hartree=-1.,gradient_hartree_per_bohr=np.ones(9),scf_state=('two','states'))
    component=SimpleNamespace(evaluate=lambda xyz,**kw:(calls.append(kw) or result),accept=lambda r:calls.append(r))
    service=object.__new__(GeometryEvaluationService);service.backend=spec;service._orbital_component=component
    point=ScanPoint(3,.1,np.zeros((3,3)))
    actual=service._evaluate_builtin_backend_with_state_following(point,tag='trial',requested_properties=('energy','gradient','hessian'))
    assert actual.point_index==3 and calls==[{'compute_hessian':True}]
    service.accept_electronic_state(SimpleNamespace(result=actual));assert calls[-1] is actual


def test_optimizer_checks_response_bytes_before_using_cache(tmp_path):
    from matrix_link.optimizer import GeometryEvaluationService
    spec=backend(tmp_path);service=object.__new__(GeometryEvaluationService)
    service._orbital_component=SimpleNamespace(backend=spec)
    # No cache or geometry attributes exist: rejection must precede their use.
    with spec.native_roks_composite_response_paths[1].open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='changed during'):service.evaluate(np.zeros(9),tag='cached')


def test_final_hessian_uses_excited_composite_backend(monkeypatch,tmp_path):
    import matrix_link.optimizer as module
    backend_spec=backend(tmp_path);calls=[]
    actual=SimpleNamespace(result=SimpleNamespace(hessian_hartree_per_bohr2=np.diag([-1.,2.,3.])))
    service=SimpleNamespace(backend=backend_spec,evaluate=lambda q,**kw:(calls.append(kw) or actual))
    def forbidden(*a,**kw):raise AssertionError('a composite analytic Hessian must not use the gradient-only fallback')
    monkeypatch.setattr(module,'_finite_difference_final_cartesian_hessian',forbidden)
    result=module._evaluate_final_cartesian_hessian(service,None,np.zeros(3),tag='final')
    assert result is actual and calls[0]['requested_properties']==('energy','gradient','hessian')


def test_restart_anchor_may_be_energy_only_but_accepted_reference_requires_gradient(monkeypatch,tmp_path):
    import json,hashlib
    import matrix_gfn2
    import matrix_gfn2.deltascf_derivatives as derivatives
    import matrix_link.composite_roks_component as adapter
    spec=backend(tmp_path);root=tmp_path/'run/roks_50_50_accepted';root.mkdir(parents=True)
    records={};called=[]
    for role in ['reference','anchor']:
        records[role]=[]
        for i in range(2):
            p=root/f'{role}_{i}.npz';p.write_bytes(b'retained state')
            records[role].append(dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),coordinates_angstrom=[[0,0,0]]))
    records.update(schema='mouse.link.roks_50_50.accepted.v1',contract=adapter._contract(spec))
    (root/'accepted.json').write_text(json.dumps(records))
    models=[SimpleNamespace(fingerprint='raw'),SimpleNamespace(fingerprint='spectral')]
    monkeypatch.setattr(adapter,'_models',lambda backend:models)
    def load(path,*,require_gradient,**kw):
        role,index=path.stem.rsplit('_',1);i=int(index);called.append((role,require_gradient))
        return SimpleNamespace(spin_reference_fingerprint=models[i].fingerprint,orbital_saddle_index=(0,1)[i],to_apoc=lambda state_id:SimpleNamespace(state_id=state_id))
    monkeypatch.setattr(derivatives,'load_derivative_point',load)
    monkeypatch.setattr(matrix_gfn2,'combine_roks_results',lambda *r,**kw:SimpleNamespace(components=r))
    component=SimpleNamespace(backend=spec,run_dir=root.parent,reference=None,anchor=None)
    adapter.restore_accepted(component)
    assert called==[('reference',True),('reference',True),('anchor',False),('anchor',False)]
    assert component.reference is not None and component.anchor is not None


def test_excited_tda_finite_difference_fallback_never_substitutes_ground_state(monkeypatch):
    from matrix_link.optimizer import OptimizerEvaluation,_finite_difference_final_cartesian_hessian
    from matrix_link.scan import PointEvaluationResult
    import matrix_link.resident_calculators as resident
    from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
    def forbidden(*a,**kw):raise AssertionError('ground-state resident fallback used for an excited state')
    monkeypatch.setattr(resident,'create_resident_calculator',forbidden)
    xyz=np.array([[0.,0.,0.],[0.,0.,1.]])
    initial=OptimizerEvaluation(np.zeros(6),xyz,PointEvaluationResult(0,0.,gradient_hartree_per_bohr=np.zeros(6)))
    calls=[]
    def evaluate(q,**kw):
        calls.append(kw);x=kw['realized_coordinates_angstrom']
        return OptimizerEvaluation(q,x,PointEvaluationResult(0,0.,gradient_hartree_per_bohr=(2*ANGSTROM_TO_BOHR*x).ravel()))
    service=SimpleNamespace(backend=QMScanBackend(name='matrix_gfn2',electronic_state=1),evaluate=evaluate)
    final=_finite_difference_final_cartesian_hessian(service,initial,np.zeros(6),tag='excited')
    assert len(calls)==12 and final.result.hessian_hartree_per_bohr2==pytest.approx(2*np.eye(6))
    assert not final.result.execution['final_cartesian_hessian_independent_displaced_scf']
