import socket
assert socket.getfqdn()=='oracle.sns.it'
import numpy as np
from response_projection import *

def symmetric_tensor(rng,n):
    a=rng.normal(size=(n,n,n,n));a=.5*(a+a.swapaxes(0,1));a=.5*(a+a.swapaxes(2,3));return .5*(a+a.transpose(2,3,0,1))

def test_exchange_algebra_and_hf_limit():
    rng=np.random.default_rng(51);v=symmetric_tensor(rng,4);t=exchange(v)
    assert np.max(abs(exchange(t)-.5*(v+t)))<1e-12
    assert np.max(abs(hamiltonian_projection(v-.5*t,-.5*t)-v))<1e-12

def test_projection_is_unique_least_squares_minimum():
    rng=np.random.default_rng(53);c=symmetric_tensor(rng,3);s=symmetric_tensor(rng,3)
    v=hamiltonian_projection(c,s);r0,r1=response_residuals(v,c,s)
    derivative=r0-.5*exchange(r0+r1)
    assert np.max(abs(derivative))<1e-12
    base=np.sum(r0*r0)+np.sum(r1*r1)
    for unused in range(10):
        direction=symmetric_tensor(rng,3)
        a,b=response_residuals(v+.1*direction,c,s)
        assert np.sum(a*a)+np.sum(b*b)>base

def test_reference_fock_is_preserved_without_an_excitation_shift():
    rng=np.random.default_rng(59);v=symmetric_tensor(rng,4);f=rng.normal(size=(4,4));f=.5*(f+f.T)
    p=np.diag([2.,2.,0.,0.]);h=normal_ordered_one_body(v,f,p)
    assert np.max(abs(fock(v,h,p)-f))<1e-12
