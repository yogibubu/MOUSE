import numpy as np
from matrix_gfn2.orbital_saddle import generalized_mode_following


def test_high_index_saddle_is_found_without_minimizing_the_energy():
    h=np.diag([-1.,-4.,2.])
    def evaluate(x,curvature=True):return .5*x@h@x,h@x,h if curvature else None
    result=generalized_mode_following(evaluate,lambda x,p:x+p,np.array([.4,-.6,.2]),saddle_index=2,maximum_iterations=30)
    assert result.converged and result.observed_index==2
    assert np.max(abs(result.point))<1e-9
    assert any(row['energy']<0 for row in result.history)


def test_stationarity_with_wrong_index_is_never_accepted():
    def objective(x,curvature=True):return .5*x@x,x,np.eye(len(x)) if curvature else None
    result=generalized_mode_following(objective,lambda x,p:x+p,np.zeros(2),saddle_index=1,maximum_iterations=3)
    assert not result.converged
    assert result.target_index==1 and result.observed_index==0


def test_minimum_of_squared_gradient_is_not_reported_as_a_stationary_state():
    # E'=1+x^2 has a strictly positive minimum at x=0, but no stationary point.
    def objective(x,curvature=True):return float(x[0]+x[0]**3/3),1+x*x,np.diag(2*x) if curvature else None
    result=generalized_mode_following(objective,lambda x,p:x+p,np.zeros(1),saddle_index=1,maximum_iterations=4)
    assert not result.converged and np.max(abs(result.gradient))>=1


def test_wrong_index_escape_does_not_exhaust_trust_before_reaching_the_barrier():
    # The maximum at x=0 lies beyond a region where climbing away from the
    # minimum x=1 increases |g|. The old geometric trust shrink stalls near .89.
    def evaluate(x,curvature=True):
        z=x[0]
        return float((z*z-1)**2),np.array([4*z*(z*z-1)]),np.array([[12*z*z-4]]) if curvature else None
    result=generalized_mode_following(evaluate,lambda x,p:x+p,np.array([.99]),
        saddle_index=1,maximum_step=.02,maximum_iterations=100)
    assert result.converged and result.observed_index==1
    assert abs(result.point[0])<1e-8
    assert max(row['residual'] for row in result.history)>result.history[0]['residual']


def test_resolved_soft_saddle_uses_a_full_step_inside_the_trust_radius():
    # A curvature floor of 1e-3 damps this certified mode by a factor 100,
    # causing linear convergence despite an exactly quadratic local surface.
    h=np.diag([-1e-5, 2.])
    def evaluate(x,curvature=True):
        return float(.5*x@h@x),h@x,h if curvature else None
    result=generalized_mode_following(evaluate,lambda x,p:x+p,
        np.array([.05,0.]),saddle_index=1,maximum_iterations=3)
    assert result.converged and result.observed_index==1
    assert result.iterations==1
    assert np.max(abs(result.point))<1e-12
