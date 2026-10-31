"""State guards must block root changes without altering stationarity criteria."""
import numpy as np
import pytest
from scipy.linalg import expm
from matrix_gfn2.roks_tracking import common_orbital_singlet_overlap,make_roks_trial_guard
from matrix_gfn2.orbital_saddle import generalized_mode_following
from matrix_gfn2 import assigned_transition_orbitals


def data():
    s=np.eye(6);c=s.copy();occ=np.zeros((2,6));occ[:,0]=1;occ[0,2]=1;occ[1,1]=1
    return s,c,occ


def test_assigned_nto_seed_preserves_ground_density_and_targets_the_full_direction():
    rng=np.random.default_rng(73)
    q,_=np.linalg.qr(rng.normal(size=(6,6)))
    s=q@np.diag(np.linspace(.6,1.4,6))@q.T
    c=q@np.diag(np.linspace(.6,1.4,6)**-.5)
    occ=np.zeros((2,6));occ[:,0]=1;occ[1,1]=1;occ[0,2]=1
    x=np.outer([.6,.8],[.5,.5,.5,.5])
    other=assigned_transition_orbitals(c,s,occ,[0,1],[2,3,4,5],x)
    np.testing.assert_allclose(other.T@s@other,np.eye(6),atol=2e-14)
    np.testing.assert_allclose(other[:,:2]@other[:,:2].T,c[:,:2]@c[:,:2].T,atol=2e-14)
    guard=make_roks_trial_guard(s,occ,other,reference_coefficients=c,
        occupied=[0,1],virtual=[2,3,4,5],reference_transition=x,
        transition_metric='rank_one',minimum_transition_overlap=.9)
    assert guard(other,other)
    assert guard.last_scores['transition']==pytest.approx(1.)
    # Changing the arbitrary ground MO chart must not change the physical CSF.
    ro=expm(np.array([[0.,.43],[-.43,0.]]))
    rv,_=np.linalg.qr(rng.normal(size=(4,4)))
    rotated=c.copy();rotated[:,:2]=c[:,:2]@ro;rotated[:,2:]=c[:,2:]@rv
    same=assigned_transition_orbitals(rotated,s,occ,[0,1],[2,3,4,5],ro.T@x@rv)
    assert common_orbital_singlet_overlap(other,same,s,occ)==pytest.approx(1.)


def test_assigned_nto_seed_rejects_degenerate_direction_and_wrong_open_labels():
    s,c,occ=data()
    with pytest.raises(ValueError,match='distinct single-pair'):
        assigned_transition_orbitals(c,s,occ,[0,1],[2,3,4,5],np.eye(2,4))
    with pytest.raises(ValueError,match='occupied hole'):
        assigned_transition_orbitals(c,s,occ,[0,2],[1,3,4,5],np.array([[1.,0,0,0],[0,0,0,0]]))


def test_fixed_transition_dominance_handles_sector_relaxation_without_accepting_competitor():
    s,c,occ=data();x=np.zeros((2,4));x[1,0]=.8;x[0,1]=.6
    guard=make_roks_trial_guard(s,occ,c,reference_coefficients=c,
        occupied=[0,1],virtual=[2,3,4,5],reference_transition=x,
        minimum_transition_overlap=np.sqrt(.5),minimum_active_overlap=0.,
        minimum_local_overlap=.9,minimum_sector_norm=.7)
    k=np.zeros((6,6));k[1,3]=k[2,0]=.65;k-=k.T
    relaxed=c@expm(k)
    assert guard(relaxed,relaxed)
    assert guard.last_scores['anchor_active']<.7
    assert guard.last_scores['raw_transition']**2>.5
    competitor=c[:,[1,0,3,2,4,5]]
    assert not guard(competitor,competitor)
    assert guard.last_scores['raw_transition']**2<.5
    # Losing the occupied/virtual sector is still rejected.
    bad=c@expm(k*2)
    assert not guard(bad,bad)


def test_guard_allows_phase_and_open_exchange_but_rejects_another_promotion():
    s,c,occ=data();guard=make_roks_trial_guard(s,occ,c)
    phase=c.copy();phase[:,1]*=-1
    assert guard(c,phase)
    swap=c[:,[0,2,1,3,4,5]]
    assert guard(c,swap)
    other=c[:,[0,3,2,1,4,5]]
    assert not guard(c,other)
    assert common_orbital_singlet_overlap(c,swap,s,occ)==pytest.approx(1.)
    # Same open subspace, different singlet dyad: subspace overlap alone is unsafe.
    k=np.zeros((6,6));k[1,2]=np.pi/4;k[2,1]=-np.pi/4
    assert not guard(c,c@expm(k))


def test_guard_uses_local_csf_but_allows_accumulated_spectator_relaxation():
    s,c,occ=data();guard=make_roks_trial_guard(s,occ,c)
    k=np.zeros((6,6));k[0,3]=1.;k[3,0]=-1.
    previous=c
    for t in np.linspace(0,1.2,25)[1:]:
        trial=c@expm(t*k)
        assert guard(previous,trial)
        previous=trial
    assert common_orbital_singlet_overlap(c,previous,s,occ)<.2
    assert not guard(c,previous)  # a single discontinuous leap is rejected


def test_fixed_reference_transition_detects_drift_across_accepted_small_steps():
    s,c,occ=data();x=np.zeros((2,4));x[1,0]=1
    # Global active gate is deliberately disabled here to isolate the fixed
    # transition gate: many locally continuous rotations may reach another root.
    guard=make_roks_trial_guard(s,occ,c,minimum_active_overlap=0.,reference_coefficients=c,occupied=[0,1],virtual=[2,3,4,5],reference_transition=x)
    k=np.zeros((6,6));k[2,3]=1.;k[3,2]=-1.
    previous=c
    for t in [.1,.2,.3,.4,.5,.6,.7]:
        trial=c@expm(t*k);assert guard(previous,trial);previous=trial
    assert not guard(previous,c@expm(.9*k))
    assert guard.last_scores['local_csf']>.9
    assert guard.last_scores['transition']<.7


def test_guard_cannot_certify_a_forbidden_stationary_point():
    def objective(x,curvature=True):return .5*float(x@x),x.copy(),np.eye(1) if curvature else None
    def retract(x,step):return x+step
    free=generalized_mode_following(objective,retract,np.array([1.]),saddle_index=0)
    guarded=generalized_mode_following(objective,retract,np.array([1.]),saddle_index=0,trial_guard=lambda old,new:new[0]>=.25)
    assert free.converged
    assert not guarded.converged
    assert guarded.point[0]>=.25 and guarded.gradient[0]>=.25


def test_fast_common_csf_overlap_matches_apoc_in_nonorthogonal_metric():
    from matrix_apoc.deltascf import DeterminantOrbitals,DeltaSCFState,SpinAdaptedSinglet,singlet_csf_overlap
    _,_,occ=data();rng=np.random.default_rng(719)
    a=rng.normal(size=(6,6));s=a@a.T+np.eye(6);w,v=np.linalg.eigh(s);c=(v*w**-.5)@v.T
    k=rng.normal(size=(6,6));d=c@expm(.2*(k-k.T))
    def csf(coeff):
        determinant=DeterminantOrbitals(np.array([4]),np.zeros((1,3)),s,(coeff,coeff),tuple(occ),np.array([0]),model='algebra-test')
        return SpinAdaptedSinglet(1,DeltaSCFState(1,determinant))
    exact,_=singlet_csf_overlap(csf(c),csf(d),s)
    assert common_orbital_singlet_overlap(c,d,s,occ)==pytest.approx(exact,abs=1e-12)


def test_rank_one_fidelity_accounts_for_multiconfigurational_reference():
    s,c,occ=data();x=np.zeros((2,4));x[1,0]=.75;x[0,1]=np.sqrt(1-.75**2)
    args=dict(reference_coefficients=c,occupied=[0,1],virtual=[2,3,4,5],reference_transition=x)
    raw=make_roks_trial_guard(s,occ,c,**args)
    ranked=make_roks_trial_guard(s,occ,c,transition_metric='rank_one',minimum_transition_overlap=.90,**args)
    k=np.zeros((6,6));k[2,4]=np.arccos(.93);k[4,2]=-k[2,4];trial=c@expm(k)
    assert not raw(c,trial)
    assert ranked(c,trial)
    assert ranked.last_scores['raw_transition']==pytest.approx(.6975)
    assert ranked.last_scores['transition']==pytest.approx(.93)
    assert ranked.last_scores['local_csf']>.9
    other=c[:,[0,1,4,3,2,5]]
    assert not ranked(c,other)


def test_rank_one_metric_rejects_ambiguous_or_unrepresentable_pair():
    s,c,occ=data();x=np.zeros((2,4));x[0,0]=x[1,1]=1/np.sqrt(2)
    with pytest.raises(ValueError,match='distinct representable'):
        make_roks_trial_guard(s,occ,c,reference_coefficients=c,occupied=[0,1],virtual=[2,3,4,5],reference_transition=x,transition_metric='rank_one')
