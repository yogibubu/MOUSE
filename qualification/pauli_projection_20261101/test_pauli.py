import socket
assert socket.getfqdn()=='oracle.sns.it'
import numpy as np
from pauli import *

def tensor(rng,n):
    v=pair_symmetric(rng.normal(size=(n,n,n,n)));return .5*(v+v.transpose(2,3,0,1))

def test_reconstruction_and_nearest_response():
    rng=np.random.default_rng(43);c=tensor(rng,4);s=tensor(rng,4);w,cp,sp,removed=project(c,s)
    assert np.max(abs(pair_symmetric(w)-.5*exchange(w)-cp))<1e-12
    assert np.max(abs(-.5*exchange(w)-sp))<1e-12
    assert np.max(abs(fully_symmetric(cp+sp)))<1e-12
    assert np.max(abs(w-w.transpose(2,3,0,1)))<1e-12
    assert np.max(abs(w-w.swapaxes(0,1).swapaxes(2,3)))<1e-12
    base=np.sum((c-cp)**2)+np.sum((s-sp)**2)
    for _ in range(5):
        _,dc,ds,_=project(tensor(rng,4),tensor(rng,4))
        assert abs(np.sum((cp-c)*dc)+np.sum((sp-s)*ds))<1e-11
        assert np.sum((cp+.1*dc-c)**2)+np.sum((sp+.1*ds-s)**2)>base

def test_hf_and_one_electron_limits():
    rng=np.random.default_rng(49);v=tensor(rng,4);k=exchange(v);w,_,_,removed=project(v-.5*k,-.5*k)
    assert np.max(abs(w-v))<1e-12 and np.max(abs(removed))<1e-12
    w,_,_,_=project(tensor(rng,4),tensor(rng,4));x=rng.normal(size=4);x/=np.linalg.norm(x);p=np.outer(x,x)
    j,k=jk(w,p);assert abs(np.sum(p*(j-k)))<1e-12

def test_nonsymmetric_integral_fci_is_hermitian_and_spin_invariant():
    from pyscf.fci import direct_nosym,spin_op
    rng=np.random.default_rng(47);n=3;w,_,_,_=project(tensor(rng,n),tensor(rng,n));h=rng.normal(size=(n,n));h=.5*(h+h.T)
    effective=direct_nosym.absorb_h1e(h,w,n,(1,1),.5);eye=np.eye(9)
    matrix=np.column_stack([direct_nosym.contract_2e(effective,x.reshape(3,3),n,(1,1)).ravel() for x in eye])
    ss=np.column_stack([spin_op.contract_ss(x.reshape(3,3),n,(1,1)).ravel() for x in eye])
    assert np.max(abs(matrix-matrix.T))<1e-12
    assert np.max(abs(matrix@ss-ss@matrix))<1e-12
    for i in range(n):
        p=np.zeros((n,n));p[i,i]=2;j,k=jk(w,p)
        expected=np.sum(h*p)+.5*np.sum(p*j)-.25*np.sum(p*k)
        assert abs(matrix[i*n+i,i*n+i]-expected)<1e-12
