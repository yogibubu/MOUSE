import socket
assert socket.getfqdn()=='oracle.sns.it'
import numpy as np
from types import SimpleNamespace
from covariant_pair import project_pair_matrix,pair_rotation
from pair_cas import evaluate_pair_cas

def test_group_projection_preserves_exact_hf_and_transforms_covariantly():
    rng=np.random.default_rng(37);n=3;no=1
    h=rng.normal(size=(n,n));h=.5*(h+h.T)
    b=rng.normal(size=(5,n,n));b=.15*(b+b.transpose(0,2,1));eri=np.einsum('kab,kcd->abcd',b,b)
    def functional(p):
        total=p.sum(0);j=np.einsum('abcd,cd->ab',eri,total);k=np.einsum('acbd,scd->sab',eri,p)
        return h+j-k,float(np.sum(h*total)+.5*np.sum(total*j)-.5*np.sum(p*k))
    ctx=SimpleNamespace(state=SimpleNamespace(overlap=np.eye(n)),fock_and_energy=functional)
    c=np.eye(n);direct=evaluate_pair_cas(ctx,c,no,hole=0,particle=no).matrix
    projected=project_pair_matrix(ctx,c,no,nodes=16)
    assert np.max(abs(projected-direct))<2e-12
    angle=.371;cc=c.copy();cc[:,0]=np.cos(angle)*c[:,0]-np.sin(angle)*c[:,no];cc[:,no]=np.sin(angle)*c[:,0]+np.cos(angle)*c[:,no]
    d=pair_rotation(angle)
    assert np.max(abs(project_pair_matrix(ctx,cc,no,nodes=16)-d.T@projected@d))<2e-12

def test_projection_removes_a_non_covariant_polynomial_matrix_term():
    # An arbitrary nonlinear density term is not assumed to define a CI Hamiltonian.
    def functional(p):
        total=p.sum(0);e=float(np.sum(total**3)+.3*np.sum(p**2))
        return 3*total**2+.6*p,e
    ctx=SimpleNamespace(state=SimpleNamespace(overlap=np.eye(2)),fock_and_energy=functional)
    c=np.eye(2);h=project_pair_matrix(ctx,c,1,nodes=16);angle=.287
    cc=np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
    d=pair_rotation(angle)
    assert np.max(abs(project_pair_matrix(ctx,cc,1,nodes=16)-d.T@h@d))<2e-12
