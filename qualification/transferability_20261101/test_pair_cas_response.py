import socket
assert socket.getfqdn()=='oracle.sns.it'
import numpy as np
from types import SimpleNamespace
from scipy.linalg import expm
from pair_cas_response import pair_cas_response

def test_pair_closure_orbital_gradient_matches_independent_energy_differences():
    rng=np.random.default_rng(29);n=4;no=2
    h=rng.normal(size=(n,n));h=.5*(h+h.T)
    b=rng.normal(size=(5,n,n));b=.15*(b+b.transpose(0,2,1));eri=np.einsum('kab,kcd->abcd',b,b)
    def j(p):return np.einsum('abcd,cd->ab',eri,p)
    def k(p):return np.einsum('acbd,...cd->...ab',eri,p)
    def functional(p):
        total=p.sum(0);jj=j(total);kk=k(p)
        return h+jj-kk,float(np.sum(h*total)+.5*np.sum(total*jj)-.5*np.sum(p*kk))
    def polarization(p):return -k(p)+.5*k(p.sum(0)),0.,None
    ctx=SimpleNamespace(state=SimpleNamespace(overlap=np.eye(n)),fock_and_energy=functional,
        density_response_kernel=lambda state:lambda dp:j(dp)-.5*k(dp),polarization=SimpleNamespace(evaluate=polarization))
    a=rng.normal(size=(n,n));c=expm(.3*(a-a.T))
    values,z,g,matrix,left,right=pair_cas_response(ctx,c,no)
    from pair_cas import evaluate_pair_cas
    assert np.max(abs(values-evaluate_pair_cas(ctx,c,no,hole=0,particle=no).energies))<1e-12
    for index,(i,jj) in enumerate(zip(left,right)):
        a=np.zeros((n,n));a[i,jj]=1;a[jj,i]=-1;eps=1e-5
        plus=evaluate_pair_cas(ctx,c@expm(eps*a),no,hole=0,particle=no).energies
        minus=evaluate_pair_cas(ctx,c@expm(-eps*a),no,hole=0,particle=no).energies
        assert np.max(abs((plus-minus)/(2*eps)-g[:,index]))<2e-9
