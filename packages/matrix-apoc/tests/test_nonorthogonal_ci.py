import numpy as np
import pytest
from matrix_apoc import NonOrthogonalSingletCIState as State,nonorthogonal_ci_overlap,follow_nonorthogonal_ci_state


def test_spin_purity_and_interference():
    e=np.eye(3);a=e[:,[0]];b=e[:,[1]]
    singlet=State(1,((a,b),(b,a)),np.ones(2)/np.sqrt(2),e)
    assert singlet.spin_squared==0
    with pytest.raises(ValueError,match="spin-pure"):
        State(1,((a,b),(b,a)),np.array([1.,-1.])/np.sqrt(2),e)
    plus=State(1,((a,a),(b,b)),np.ones(2)/np.sqrt(2),e)
    minus=State(2,((a,a),(b,b)),np.array([1.,-1.])/np.sqrt(2),e)
    assert nonorthogonal_ci_overlap(plus,minus,e)[0]<1e-14
    selected,match=follow_nonorthogonal_ci_state(plus,[minus,plus],e)
    assert selected is plus and match.overlap>1-1e-14


def test_nonorthogonal_metric_and_ao_covariance():
    e=np.eye(3);a=e[:,[0]];b=np.array([[.6],[.8],[0.]])
    norm=np.sqrt(2+2*.6**2)
    state=State(1,((a,a),(b,b)),np.ones(2)/norm,e)
    t=np.array([[1.,.2,0],[0,1.2,.1],[0,0,.8]])
    inv=np.linalg.inv(t);s=t.T@t
    transformed=State(1,tuple((inv@x,inv@y) for x,y in state.determinants),state.coefficients,s)
    assert nonorthogonal_ci_overlap(state,transformed,t)[0]>1-1e-14


def test_exact_four_spin_singlets():
    from matrix_gfn2.state_interaction import four_open_shell_singlets
    basis=four_open_shell_singlets(np.eye(4),2)
    for i,csf in enumerate(basis):
        state=State(i+1,tuple(pair for weight,pair in csf),np.array([weight for weight,pair in csf]),np.eye(4))
        assert state.spin_squared==0
