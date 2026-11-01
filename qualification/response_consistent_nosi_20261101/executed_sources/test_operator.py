"""Independent scalar-CI and explicit-A checks of contracted response."""
from types import SimpleNamespace
import numpy as np
from matrix_gfn2.state_interaction import Hamiltonian
from transition import vectors_in_frame
from response_operator import projected_response

def test_transition_vectors_and_HF_operator_against_literal_CIS():
    rng=np.random.default_rng(219);n=5;no=2;c=np.linalg.qr(rng.normal(size=(n,n)))[0]
    eri=rng.normal(size=(n,n,n,n));eri=.5*(eri+eri.swapaxes(0,1));eri=.5*(eri+eri.swapaxes(2,3));eri=.5*(eri+eri.transpose(2,3,0,1));h=rng.normal(size=(n,n));h=.5*(h+h.T)
    levels=np.arange(n)*.7;f=(c*levels)@c.T;ham=Hamiltonian(h,eri);g=c[:,:no];ground=(g,g);basis=[]
    for i in range(no):
        for a in range(no,n):
            first=c[:,[j for j in range(no) if j!=i]+[a]];basis.append([(2**-.5,(first,g)),(2**-.5,(g,first))])
    t=vectors_in_frame(c,basis,no);np.testing.assert_allclose(t@t.T,np.eye(len(t)),atol=1e-12)
    class Kernel:
        def __call__(self,p):return .3*p+.2*np.trace(p)*np.eye(n)
        def response_batch(self,p):return np.array([self(x) for x in p])
    context=SimpleNamespace(fock_and_energy=lambda p:(np.array([f,f]),0.),density_response_kernel=lambda s:Kernel())
    model,hf=projected_response(context,SimpleNamespace(),c,levels,no,t,eri,.7*eri,h)
    eg=ham.element(ground,ground)[1];reference=np.array([[ham.csf(a,b)[1]-eg*ham.csf(a,b)[0] for b in basis] for a in basis]);np.testing.assert_allclose(hf,reference,atol=1e-10)
    co=c[:,:no];cv=c[:,no:];ii=np.repeat(np.arange(no),n-no);aa=np.tile(np.arange(no,n),no);static=np.diag(levels[aa]-levels[ii])
    for j,(i,a) in enumerate(zip(ii,aa)):
        p=np.outer(c[:,i],c[:,a]);v=Kernel()(.5*(p+p.T));static[:,j]+=2*np.array([c[:,k]@v@c[:,b] for k,b in zip(ii,aa)])
    exchange=.9*(.19+.46*.7)*eri
    direct=np.einsum('mnpq,mi,nj,pa,qb->iajb',exchange,co,co,cv,cv,optimize=True).reshape(len(t),len(t))
    crossed=np.einsum('mnpq,mi,nb,pj,qa->iajb',exchange,co,cv,co,cv,optimize=True).reshape(len(t),len(t))
    np.testing.assert_allclose(model,t@(static+.5*(crossed-direct))@t.T,atol=1e-10)
