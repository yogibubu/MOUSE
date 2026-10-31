"""Tracking must tolerate spectator relaxation and reject active-root changes."""
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
import pytest
from scipy.linalg import expm
from matrix_gfn2.roks_tracking import active_orbital_character


def example():
    # Four closed/virtual spectator rotations can suppress determinant overlap
    # without changing the excitation's two active orbitals.
    n=10;xyz=np.array([[0.,0.,0.],[1.3,0.,0.],[-.4,1.1,0.]])
    state=SimpleNamespace(atomic_numbers=np.array([6,1,1]),coordinates_bohr=xyz,
        ao_atom=np.array([0]*4+[1]*3+[2]*3),ao_shell=['2s']+['2p']*3+['2p']*3+['2p']*3,model_id='test',charge=0)
    occ=np.zeros((2,n));occ[:,[0,4,6,8]]=1;occ[0,1]=1;occ[1,2]=1
    return SimpleNamespace(reference_state=state,overlap=np.eye(n),common_coefficients=np.eye(n),occupations=occ,
        core_electrons=np.zeros(3),spin_reference_fingerprint='frozen')


def test_closed_shell_relaxation_does_not_change_active_character():
    a=example();b=SimpleNamespace(**vars(a));g=np.zeros((10,10))
    for i,j in [(0,3),(4,5),(6,7),(8,9)]:g[i,j]=1.;g[j,i]=-1.
    b.common_coefficients=expm(g)
    assert abs(np.linalg.det(b.common_coefficients[np.ix_([0,4,6,8],[0,4,6,8])]))<.1
    assert active_orbital_character(a,b,a.reference_state.coordinates_bohr)['transition']==pytest.approx(1.)


def test_rigid_rotation_translation_and_phases_are_invariant():
    a=example();b=SimpleNamespace(**vars(a));state=SimpleNamespace(**vars(a.reference_state))
    g=np.array([[0.,.6,-.2],[-.6,0.,.3],[.2,-.3,0.]])
    r=expm(g);u=np.eye(10)
    for ids in [[1,2,3],[4,5,6],[7,8,9]]:u[np.ix_(ids,ids)]=r
    state.coordinates_bohr=state.coordinates_bohr@r.T+np.array([3.,-2.,1.]);b.reference_state=state
    b.common_coefficients=u.copy();b.common_coefficients[:,1]*=-1;b.overlap=u@a.overlap@u.T
    assert active_orbital_character(a,b,a.reference_state.coordinates_bohr)['transition']==pytest.approx(1.)


def test_active_orbital_change_is_rejected_by_character():
    a=example();b=SimpleNamespace(**vars(a));b.common_coefficients=np.eye(10)[:,[0,3,2,1,4,5,6,7,8,9]]
    assert active_orbital_character(a,b,a.reference_state.coordinates_bohr)['transition']==0.
    b.common_coefficients=np.eye(10)[:,[0,2,1,3,4,5,6,7,8,9]]
    assert active_orbital_character(a,b,a.reference_state.coordinates_bohr)['transition']==pytest.approx(1.)
    g=np.zeros((10,10));g[1,2]=np.pi/4;g[2,1]=-np.pi/4
    b.common_coefficients=expm(g)
    assert active_orbital_character(a,b,a.reference_state.coordinates_bohr)['transition']<1e-12


@pytest.mark.parametrize('field,value',[('spin_reference_fingerprint','other'),('occupations',np.zeros((2,10)))])
def test_incompatible_state_rejected(field,value):
    a=example();b=SimpleNamespace(**vars(a));setattr(b,field,value)
    with pytest.raises(ValueError,match='same model'):active_orbital_character(a,b,a.reference_state.coordinates_bohr)


def test_following_policy_is_explicit_and_changes_restart_contract(tmp_path):
    from test_composite_roks_link import backend
    from matrix_link.composite_roks_component import _contract,_allowed_index
    from matrix_link.scan import _composite_restart_json_value
    a=backend(tmp_path);b=replace(a,native_roks_composite_following='character');c=replace(a,native_roks_composite_following='character_adaptive_index')
    assert len({_contract(x) for x in [a,b,c]})==3
    assert 'native_roks_composite_following' not in _composite_restart_json_value(a)
    assert _composite_restart_json_value(b)['native_roks_composite_following']=='character'
    assert not _allowed_index(b,1,0) and _allowed_index(c,1,0) and not _allowed_index(c,2,0)
    with pytest.raises(ValueError):replace(a,native_roks_composite_following='unknown')
    with pytest.raises(ValueError):replace(a,native_excited_method='tda',deltascf_promotion=None,native_roks_composite_response_paths=None,native_roks_composite_response_sha256=None,native_roks_composite_saddle_indices=None,native_roks_composite_following='character')


def test_adjacent_index_retry_is_bounded_and_uses_accepted_seed(monkeypatch,tmp_path):
    from test_composite_roks_link import backend
    import matrix_gfn2, matrix_gfn2.roks as roks
    from matrix_link.composite_roks_component import _evaluate_components
    spec=replace(backend(tmp_path),native_roks_composite_following='character_adaptive_index')
    old=[SimpleNamespace(orbital_saddle_index=0),SimpleNamespace(orbital_saddle_index=1)]
    calls=[]
    def evaluate(*a,**kw):
        calls.append(kw)
        if len(calls)==1:raise roks.ROKSConvergenceError(SimpleNamespace(orbital_saddle_index=1,target_orbital_saddle_index=0,orbital_residual=1e-7))
        return SimpleNamespace(orbital_saddle_index=kw['saddle_index'])
    monkeypatch.setattr(roks,'evaluate_gfn2_roks_singlet',evaluate)
    monkeypatch.setattr(matrix_gfn2,'combine_roks_results',lambda *r,**kw:r)
    component=SimpleNamespace(backend=spec,calculator=SimpleNamespace(_calculator=object()))
    result,changes=_evaluate_components(component,None,None,None,['raw','spectral'],SimpleNamespace(components=old),None)
    assert [c['saddle_index'] for c in calls]==[0,1,1] and len(changes)==1
    assert calls[0]['reference_result'] is calls[1]['reference_result'] is old[0]
    calls.clear()
    component.backend=replace(spec,native_roks_composite_following='character')
    with pytest.raises(roks.ROKSConvergenceError):_evaluate_components(component,None,None,None,['raw','spectral'],SimpleNamespace(components=old),None)
    assert len(calls)==1


def test_character_policy_keeps_local_csf_gate(monkeypatch,tmp_path):
    from test_composite_roks_link import backend
    import matrix_link.composite_roks_component as adapter
    import matrix_gfn2.roks_tracking as tracking
    import matrix_apoc
    spec=replace(backend(tmp_path),native_roks_composite_following='character')
    states=[SimpleNamespace(reference_state=object(),orbital_residual=0.),SimpleNamespace(reference_state=object(),orbital_residual=0.)]
    result=SimpleNamespace(components=states,energy_hartree=-1.,gradient_hartree_per_bohr=np.zeros((3,3)),component_csf_overlap=.99,component_active_pair_overlap=None,component_tracking="csf",orbital_saddle_indices=(0,1))
    pair=(result,('a','b'));c=SimpleNamespace(backend=spec,reference=pair,anchor=pair,run_dir=tmp_path/'run',calculator=SimpleNamespace(_calculator=object()),_cross_overlap=lambda *a:None)
    monkeypatch.setattr(adapter,'_models',lambda *a:[SimpleNamespace(anchor_coordinates_bohr=None,fingerprint='a')]*2)
    monkeypatch.setattr(adapter,'_evaluate_components',lambda *a:(result,[]));monkeypatch.setattr(adapter,'_selected',lambda *a:('c','d'))
    monkeypatch.setattr(tracking,'active_orbital_character',lambda *a:dict(transition=.95))
    def scores(values):
        seq=iter(values);monkeypatch.setattr(matrix_apoc,'singlet_csf_overlap',lambda *a:(next(seq),None))
    ground=SimpleNamespace(orbital_state=None)
    # Local continuity passes, old determinant product falls below threshold.
    result.components[0].occupations=np.zeros((2,6))
    scores([.99,.98,.3,.2])
    point=adapter.evaluate_composite_component(c,np.zeros((3,3)),False,ground,{})
    assert point.execution['state_anchor_overlap']==.2
    assert point.execution['anchor_active_orbital_character'][0]['transition']==.95
    scores([.69])
    with pytest.raises(matrix_apoc.StateContinuityError,match='reference CSF'):adapter.evaluate_composite_component(c,np.zeros((3,3)),False,ground,{})
    scores([.99,.98,.3,.2]);monkeypatch.setattr(tracking,'active_orbital_character',lambda *a:dict(transition=.69))
    with pytest.raises(matrix_apoc.StateContinuityError,match='active-orbital'):adapter.evaluate_composite_component(c,np.zeros((3,3)),False,ground,{})
    assert len(list((c.run_dir/'rejected_trials').glob('*.json')))==2
    assert c.reference is pair and c.anchor is pair
    c.backend=replace(spec,native_roks_composite_following=None);scores([.99,.98,.3])
    with pytest.raises(matrix_apoc.StateContinuityError,match='anchor CSF'):adapter.evaluate_composite_component(c,np.zeros((3,3)),False,ground,{})


def test_ground_recovery_is_explicit_and_only_uses_accepted_reference(monkeypatch,tmp_path):
    from test_composite_roks_link import backend
    from matrix_link.qmqm_component import CappedQMComponent
    from matrix_link.composite_roks_component import _contract
    import matrix_gfn2.response_reference as ref
    spec=backend(tmp_path);enabled=replace(spec,native_roks_composite_ground_recovery='accepted_retry')
    assert _contract(enabled)!=_contract(spec)
    accepted=object();calls=[]
    def stationary(calc,xyz,**kw):
        calls.append(kw)
        if not kw:raise RuntimeError('cold reference failed')
        assert kw['seed_state'] is accepted
        return 'ground',{'gate':'unchanged'}
    monkeypatch.setattr(ref,'stationary_response_reference',stationary)
    c=CappedQMComponent(enabled,SimpleNamespace(_calculator=SimpleNamespace()),tmp_path/'component')
    c.reference=(SimpleNamespace(components=[SimpleNamespace(reference_state=accepted)]),None)
    c._native_excited=lambda xyz,h,ground,diagnostics:(ground,diagnostics)
    ground,diagnostics=c.evaluate(np.zeros((3,3)))
    assert len(calls)==2 and ground=='ground' and diagnostics['gate']=='unchanged'
    c.backend=spec;calls.clear()
    with pytest.raises(RuntimeError,match='cold reference'):c.evaluate(np.zeros((3,3)))
    assert calls==[{}]


def test_component_disagreement_is_a_rejected_trial_not_optimizer_abort(monkeypatch,tmp_path):
    from test_composite_roks_link import backend
    from matrix_link.composite_roks_component import _evaluate_components
    from matrix_gfn2.composite_roks import CompositeROKSStateMismatch
    from matrix_apoc import StateContinuityError
    import matrix_gfn2, matrix_gfn2.roks as roks
    monkeypatch.setattr(roks,'evaluate_gfn2_roks_singlet',lambda *a,**kw:object())
    def mismatch(*a,**kw):raise CompositeROKSStateMismatch('component overlap below threshold')
    monkeypatch.setattr(matrix_gfn2,'combine_roks_results',mismatch)
    c=SimpleNamespace(backend=backend(tmp_path),calculator=SimpleNamespace(_calculator=object()))
    with pytest.raises(StateContinuityError,match='component overlap'):_evaluate_components(c,None,None,None,['raw','spectral'],None,None)
