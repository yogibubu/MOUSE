"""Frozen-model identity and analytic derivative dispatch through LINK."""
from dataclasses import asdict,replace
from types import SimpleNamespace
import numpy as np
import pytest
from matrix_gfn2 import FrozenResponseReference,FrozenROKSAnchor,save_frozen_roks_anchor
from matrix_link.scan import QMScanBackend


def backend(tmp_path):
    ref=FrozenResponseReference(np.array([1,1]),np.arange(2),-.1*np.eye(2),'test',.9,
                               .03*np.eye(2),np.ones(2))
    a=FrozenROKSAnchor(ref,np.array([[0,1],[1,0]]),np.zeros((2,3)),.3,.2,'test',kind='relaxed_vertical')
    path=tmp_path/'anchor.npz';save_frozen_roks_anchor(path,a)
    b=QMScanBackend(name='matrix_gfn2',electronic_state=1,native_excited_method='roks',
        deltascf_promotion=(0,1),orbital_saddle_index=1,native_roks_anchor_path=path)
    return b,a


def test_backend_roundtrip_and_midcalculation_file_change_are_guarded(tmp_path):
    from matrix_link.deltascf_component import evaluate_component
    from matrix_link.scan import _composite_restart_json_value
    from matrix_link.optimizer import _restart_identity_value
    b,a=backend(tmp_path)
    assert QMScanBackend(**asdict(b)).native_roks_anchor_sha256==b.native_roks_anchor_sha256
    legacy=_composite_restart_json_value(QMScanBackend(name='matrix_gfn2'))
    assert 'native_roks_anchor_path' not in legacy and 'native_roks_anchor_sha256' not in legacy
    optimizer_legacy=_restart_identity_value(QMScanBackend(name='matrix_gfn2'))
    assert 'native_roks_anchor_path' not in optimizer_legacy and 'native_roks_anchor_sha256' not in optimizer_legacy
    assert _composite_restart_json_value(b)['native_roks_anchor_sha256']==b.native_roks_anchor_sha256
    with pytest.raises(ValueError):replace(b,native_excited_method='tda',deltascf_promotion=None,orbital_saddle_index=None)
    with b.native_roks_anchor_path.open('ab') as stream:stream.write(b'changed')
    c=SimpleNamespace(backend=b,calculator=SimpleNamespace(_calculator=None))
    with pytest.raises(ValueError,match='changed during'):
        evaluate_component(c,np.zeros((2,3)),True,None,{})


def test_link_dispatches_both_energy_gradient_and_analytic_hessian_to_one_anchor(tmp_path,monkeypatch):
    from matrix_link.deltascf_component import evaluate_component
    import matrix_gfn2
    b,a=backend(tmp_path);calls=[]
    result=SimpleNamespace(occupations=a.occupations,energy_hartree=-1.,
        gradient_hartree_per_bohr=np.ones((2,3))*.01,spin_squared=0.,orbital_residual=0.,orbital_saddle_index=1,
        to_apoc=lambda i:SimpleNamespace(state_id=i))
    def energy(calc,coords,anchor,**options):
        calls.append(('energy',anchor.fingerprint,options));return result
    def hessian(calc,coords,anchor,**options):
        calls.append(('hessian',anchor.fingerprint,options));return SimpleNamespace(hessian_hartree_per_bohr2=np.eye(6))
    monkeypatch.setattr(matrix_gfn2,'evaluate_anchored_roks',energy)
    monkeypatch.setattr(matrix_gfn2,'evaluate_anchored_roks_analytic_hessian',hessian)
    c=SimpleNamespace(backend=b,calculator=SimpleNamespace(_calculator=None),reference=None,anchor=None)
    state=SimpleNamespace(alpha_occupations=np.array([1.,0.]),beta_occupations=np.array([1.,0.]))
    point=evaluate_component(c,np.zeros((2,3)),True,SimpleNamespace(orbital_state=state),{})
    assert [x[0] for x in calls]==['energy','hessian']
    assert all(x[1]==a.fingerprint for x in calls)
    assert point.execution['hessian_source']=='analytic'
    assert point.execution['frozen_roks_anchor_fingerprint']==a.fingerprint
    assert point.hessian_hartree_per_bohr2==pytest.approx(np.eye(6))
