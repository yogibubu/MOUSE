import socket
assert socket.getfqdn()=='oracle.sns.it'
import sys
sys.path.insert(0,'/home/enzo/MOUSE-nosi-20261101')
import numpy as np
from fast_slater import Hamiltonian
from slater_condon import element
from spin_four import *

def test_fast_element_singular_and_regular():
    rng=np.random.default_rng(113);n=6;h=rng.normal(size=(n,n));h=.5*(h+h.T);w=rng.normal(size=(n,n,n,n));w=.5*(w+w.swapaxes(0,1));w=.5*(w+w.swapaxes(2,3));w=.5*(w+w.transpose(2,3,0,1));ham=Hamiltonian(h,w)
    for rank in [0,1,2,3]:
        a=np.eye(n)[:,:3];b=np.eye(n)[:,list(range(rank,3))+list(range(3,3+rank))]
        assert np.max(abs(np.array(ham.element((a,a),(b,b)))-element((a,a),(b,b),h,w)))<1e-11
    a=np.linalg.qr(rng.normal(size=(n,n)))[0][:,:3];b=np.linalg.qr(rng.normal(size=(n,n)))[0][:,:3]
    assert np.max(abs(np.array(ham.element((a,a),(b,b)))-element((a,a),(b,b),h,w)))<1e-11

def test_four_spin_exact_singlets_and_high_spin_completion():
    from pyscf.fci import cistring,spin_op
    vectors,phase=spin_vectors();assert np.max(abs(vectors.T@vectors-np.eye(2)))<1e-12
    strings=cistring.make_strings(range(4),2);ci=np.zeros((6,6,2))
    for row,alpha in enumerate(ASSIGNMENTS):
        beta=[i for i in range(4) if i not in alpha];ia=list(strings).index(sum(1<<i for i in alpha));ib=list(strings).index(sum(1<<i for i in beta));ci[ia,ib]=phase[row]*vectors[row]
    for i in range(2):assert np.max(abs(spin_op.contract_ss(ci[:,:,i],4,(2,2))))<1e-12
    rng=np.random.default_rng(37);m=rng.normal(size=(4,4));m=.5*(m+m.T)
    # For isotropic all-pair coupling the entire S=0 subspace is degenerate.
    isotropic=np.full((4,4),.2);np.fill_diagonal(isotropic,.3)
    assert np.max(abs(matrix(1.,isotropic)-np.eye(2)*.4))<1e-12
