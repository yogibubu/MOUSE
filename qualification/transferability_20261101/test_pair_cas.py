"""Independent FCI singlet and orbital-invariance limits for the pair closure."""
from pathlib import Path
import sys,socket
assert socket.getfqdn()=='oracle.sns.it'
sys.path.insert(0,'/home/enzo/MOUSE-orbital-performance-20261031/multicsf_integration/final_validation')
from common import *
from pair_cas import evaluate_pair_cas
from pyscf import fci

def test_hf_pair_cas_matches_full_spin_projected_ci():
 rng=np.random.default_rng(7543);h=rng.normal(size=(2,2));h=.5*(h+h.T)
 b=rng.normal(size=(5,2,2));b=.2*(b+b.transpose(0,2,1));eri=np.einsum('kab,kcd->abcd',b,b)
 def functional(p):
  total=p.sum(0);j=np.einsum('abcd,cd->ab',eri,total);k=np.einsum('acbd,scd->sab',eri,p)
  f=h+j-k;e=np.sum(h*total)+.5*np.sum(total*j)-.5*np.sum(p*k)
  return f,float(e)
 ctx=SimpleNamespace(state=SimpleNamespace(overlap=np.eye(2)),fock_and_energy=functional)
 eff=fci.direct_spin1.absorb_h1e(h,eri,2,(1,1),.5);unit=np.eye(4).reshape(4,2,2)
 ham=np.array([fci.direct_spin1.contract_2e(eff,v,2,(1,1)).ravel() for v in unit]).T
 s2=np.array([fci.spin_op.contract_ss(v,2,(1,1)).ravel() for v in unit]).T
 ws,vs=np.linalg.eigh(s2);singlets=vs[:,abs(ws)<1e-10];assert singlets.shape==(4,3)
 exact=np.linalg.eigvalsh(singlets.T@ham@singlets)
 for angle in [0.,.31,.79]:
  c=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
  result=evaluate_pair_cas(ctx,c,1)
  assert np.max(abs(result.energies-exact))<2e-12
