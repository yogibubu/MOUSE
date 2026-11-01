import socket
assert socket.getfqdn()=='oracle.sns.it'
import numpy as np
from trust_kkt import residual_trust_step

def test_stiff_residual_is_not_suppressed_by_soft_mode_clipping():
    jac=np.diag([1e-8,1.]);res=np.array([1e-8,1.]);radius=.25
    step=residual_trust_step(jac,res,np.ones(2),radius)
    assert np.linalg.norm(step)<=radius+1e-12
    assert np.linalg.norm(jac@step+res)<.75000001

def test_coordinate_scaling_and_redundant_constraint():
    jac=np.array([[2.,0.,0.],[0.,3.,0.],[0.,0.,0.]])
    res=np.array([.2,.3,0.]);scale=np.array([4.,2.,1.])
    step=residual_trust_step(jac,res,scale,1.)
    assert np.max(abs(jac@step+res))<1e-12
    assert np.linalg.norm(scale*step)<=1.
    assert abs(step[2])<1e-12
