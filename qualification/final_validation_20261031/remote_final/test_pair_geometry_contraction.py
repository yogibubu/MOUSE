"""Independent literal operator-jet reference for contracted nuclear response."""
from common import *
from matrix_gfn2.nonlocal_response import PairTerm,TransportedOperators

def literal(term,p,ds,d2s):
 da=term.transport.first();db=term.metric(ds[:,None],term.a)+term.metric(term.s,da)
 dq=np.einsum('raij,ij->ra',db,p);v=term.field(p)
 df=np.einsum('a,raij->rij',v,db)+np.einsum('ra,aij->rij',dq@term.k.T,term.b)
 rr=dq@term.k@dq.T
 for i in range(len(ds)):
  for j in range(len(ds)):
   d2b=(term.metric(d2s[i,j],term.a)+term.metric(ds[i],da[j])+term.metric(ds[j],da[i])+term.metric(term.s,term.transport.second(i,j)))
   rr[i,j]+=np.einsum('a,aij,ij',v,d2b,p)
 gradient=np.einsum('a,raij,ij->r',v,term.metric(term.s,da),p)
 return df,rr,gradient

def test_density_pullback_matches_literal_feature_jets():
 rng=np.random.default_rng(7543);n=12;nr=9;m=17
 sym=lambda a:.5*(a+a.swapaxes(-1,-2))
 xyz=np.array([[0.,0.,0.],[1.7,.1,.05],[.2,1.5,-.1]])
 state=SimpleNamespace(atomic_numbers=np.array([6,6,8]),ao_atom=np.repeat(np.arange(3),4),ao_shell=['2s','2p','2p','2p']*3,coordinates_bohr=xyz+.02*rng.normal(size=(3,3)),overlap=np.eye(n)+sym(.01*rng.normal(size=(n,n))))
 operators=sym(.1*rng.normal(size=(m,n,n)));k=sym(.1*rng.normal(size=(m,m)));p=sym(.1*rng.normal(size=(n,n)))
 ds=sym(.01*rng.normal(size=(nr,n,n)));d2s=sym(.01*rng.normal(size=(nr,nr,n,n)));d2s=.5*(d2s+d2s.swapaxes(0,1))
 term=PairTerm(TransportedOperators(state,xyz,operators),k,.1*rng.normal(size=m),.1*rng.normal(size=m))
 df,rr=term.geometry(p,ds,d2s);gradient=term.explicit_gradient(p).ravel()
 assert term.transport._first is None  # Production avoids the feature-jet tensor.
 old_df,old_rr,old_g=literal(term,p,ds,d2s)
 assert np.max(abs(df-old_df))<2e-13
 assert np.max(abs(rr-old_rr))<2e-13
 assert np.max(abs(gradient-old_g))<2e-13
 # Independent finite differences of the fixed-density Fock and energy.
 direction=rng.normal(size=nr);direction/=np.linalg.norm(direction);eps=2e-4
 values=[];focks=[]
 for sign in [-1,1]:
  t=sign*eps;overlap=state.overlap+t*np.einsum('r,rij->ij',direction,ds)+.5*t*t*np.einsum('r,s,rsij->ij',direction,direction,d2s)
  displaced=SimpleNamespace(**{**vars(state),'coordinates_bohr':state.coordinates_bohr+t*direction.reshape(-1,3),'overlap':overlap})
  trial=PairTerm(TransportedOperators(displaced,xyz,operators),k,term.q0,term.v);f,e=trial.evaluate(p);focks.append(f);values.append(e)
 assert np.max(abs((focks[1]-focks[0])/(2*eps)-np.einsum('r,rij->ij',direction,df)))<2e-8
 numerical=(values[1]+values[0]-2*term.evaluate(p)[1])/eps**2
 assert abs(numerical-direction@rr@direction)<2e-7

def test_batched_response_and_reused_orbital_block_preserve_nuclear_blocks():
 from matrix_gfn2.deltascf_analytic_hessian import _response_blocks
 rng=np.random.default_rng(49);n=5;nr=4
 sym=lambda a:.5*(a+a.swapaxes(-1,-2))
 c=np.linalg.qr(rng.normal(size=(n,n)))[0];occ=np.array([[1,1,0,0,0],[1,0,1,0,0]])
 fock=sym(rng.normal(size=(2,n,n)));ds=sym(.01*rng.normal(size=(nr,n,n)));d2s=sym(.01*rng.normal(size=(nr,nr,n,n)));d2s=.5*(d2s+d2s.swapaxes(0,1))
 df=sym(rng.normal(size=(nr,n,n)));rr=sym(rng.normal(size=(nr,nr)));matrix=sym(rng.normal(size=(n*n,n*n)))
 class Kernel:
  def __call__(self,p):return (matrix@p.ravel()).reshape(n,n)
  def response_batch(self,p):return (p.reshape(len(p),-1)@matrix.T).reshape(p.shape)
 kernel=Kernel();polarization=SimpleNamespace(orbital_shell=np.arange(n),coupling_hartree=np.zeros((n,n)),overlap=np.eye(n),evaluate=lambda p:(np.zeros_like(p),0.,None),response_batch=lambda p:np.zeros_like(p))
 args=([c],[(1.,occ,fock)],ds,d2s,[(df,rr)])
 scalar=_response_blocks(*args,[lambda p:kernel(p)],polarization)
 batched=_response_blocks(*args,[kernel],polarization)
 reduced=_response_blocks(*args,[kernel],polarization,compute_orbital_hessian=False)
 for a,b in zip(scalar,batched):assert np.max(abs(a-b))<2e-12
 assert reduced[1] is None
 assert np.max(abs(reduced[0]-batched[0]))<2e-12
 assert np.max(abs(reduced[2]-batched[2]))<2e-12
