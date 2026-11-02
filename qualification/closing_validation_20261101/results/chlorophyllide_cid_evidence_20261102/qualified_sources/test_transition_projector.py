import numpy as np
from matrix_gfn2._transition_projector import constrained_vertices
from matrix_gfn2._nosi_slater import transition_density

def test_gap_constraint_matches_direct_constrained_resolvent_and_is_basis_invariant():
 rng=np.random.default_rng(418);l=rng.normal(size=(6,12));gap=np.linspace(.2,2,12);t=rng.normal(size=(12,3));ids=[0,5];a=np.column_stack((np.eye(12)[:,ids],t));d=1/gap
 expected=4*l@(np.diag(d)-(d[:,None]*a)@np.linalg.solve(a.T@(d[:,None]*a),(d[:,None]*a).T))@l.T
 b,certificate=constrained_vertices(2*l/np.sqrt(gap),gap,ids,t)
 np.testing.assert_allclose(b@b.T,expected,atol=1e-12)
 rotation=np.linalg.qr(rng.normal(size=(3,3)))[0];other,_=constrained_vertices(2*l/np.sqrt(gap),gap,ids,t@rotation)
 np.testing.assert_allclose(other@other.T,b@b.T,atol=1e-12);assert certificate['total_excluded_directions']==5

def test_one_body_density_recovers_single_and_rejects_double_promotion():
 y=np.eye(6);ground=[(1.,(y[:,:3],y[:,:3]))]
 single=[(2**-.5,(y[:,[0,1,3]],y[:,:3])),(2**-.5,(y[:,:3],y[:,[0,1,3]]))]
 gamma=transition_density(ground,single);assert abs(gamma[3,2]-np.sqrt(2))<1e-12
 pair=[(1.,(y[:,[0,1,3]],y[:,[0,1,3]]))];np.testing.assert_allclose(transition_density(ground,pair),0.,atol=1e-12)
