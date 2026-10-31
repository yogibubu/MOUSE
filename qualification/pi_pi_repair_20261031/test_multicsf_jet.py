"""Independent Hamiltonian and finite-rotation checks; run on a compute host."""
import socket
assert socket.getfqdn()=='oracle.sns.it'
import os,sys
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',MOUSE_VALIDATION_ROOT='/home/enzo/MOUSE-orbital-performance-20261031')
sys.path.insert(0,'/home/enzo/MOUSE-orbital-performance-20261031/source/qualification')
import remote_bootstrap
import numpy as np
from scipy.linalg import expm
from types import SimpleNamespace
from multicsf_objective import MultiCSFObjective,quartic_jet
from pyscf import fci
rng=np.random.default_rng(17031);n=5;no=2
b=rng.normal(size=(8,n,n));b=.5*(b+b.transpose(0,2,1));eri=np.einsum('xij,xkl->ijkl',b,b)*.02
h=np.diag([-1.,-.7,.2,.5,.9])+rng.normal(size=(n,n))*.01;h=.5*(h+h.T)
def j(p):return np.einsum('ijkl,kl->ij',eri,p)
def k(p):return np.einsum('ikjl,kl->ij',eri,p)
def functional(p):
 f=np.array([h+j(p.sum(0))-k(x) for x in p]);e=sum(np.sum(x*(h+y))*.5 for x,y in zip(p,f));return f,float(e)
class Spin:
 def evaluate(self,p):
  v=-.5*k(p[0]-p[1]);return np.array([v,-v]),0.,None
ctx=SimpleNamespace(state=SimpleNamespace(overlap=np.eye(n)),fock_and_energy=functional,density_response_kernel=lambda s:lambda p:j(p)-.5*k(p),polarization=Spin())
amp=np.zeros((no,n-no));amp[0,0]=.8;amp[1,1]=.6
objective=MultiCSFObjective(ctx,eri,no,2,np.eye(n),amp);c=expm((rng.normal(size=(n,n))-rng.normal(size=(n,n)).T)*0.)
kappa=rng.normal(size=(n,n))*.04;kappa=kappa-kappa.T;c=expm(kappa)
e,g,H=objective.evaluate(c);matrix=objective.cache['matrix'].copy();coeff=objective.cache['ci'].copy()
# Compare to an explicitly contracted fermionic Hamiltonian, independent of
# orbital derivative formulas and of the 2J-K coupling expression.
hmo=c.T@h@c;gmo=np.einsum('ijkl,ip,jq,kr,ls->pqrs',eri,c,c,c,c,optimize=True)
heff=fci.direct_spin1.absorb_h1e(hmo,gmo,n,(no,no),.5);nd=fci.cistring.num_strings(n,no);basis=[]
for i in range(2):
 o=objective.occupations[i];v=np.zeros((nd,nd));phase=(-1)**(no-1+i)
 for occupations in [o,o[::-1]]:
  bits=[sum(1<<int(x) for x in np.flatnonzero(a)) for a in occupations];addresses=[fci.cistring.str2addr(n,no,x) for x in bits];v[tuple(addresses)]+=phase/np.sqrt(2)
 assert abs(fci.spin_op.spin_square(v,n,(no,no))[0])<1e-12
 basis.append(v)
exact=np.array([[np.sum(a*fci.direct_spin1.contract_2e(heff,b,n,(no,no))) for b in basis] for a in basis]);assert np.max(abs(exact-matrix))<1e-12,(exact,matrix)
for direction_id in range(4):
 direction=rng.normal(size=len(g));direction/=np.linalg.norm(direction);rot=np.zeros((n,n));rot[objective.left,objective.right]=direction;rot[objective.right,objective.left]=-direction
 eps=2e-4
 ep=objective.evaluate(c@expm(eps*rot),False)[0];em=objective.evaluate(c@expm(-eps*rot),False)[0]
 assert abs((ep-em)/(2*eps)-g@direction)<2e-7
 assert abs((ep-2*e+em)/eps**2-direction@H@direction)<2e-6,((ep-2*e+em)/eps**2,direction@H@direction)
assert abs(objective.signed_overlap(c,coeff,c,coeff)-1)<1e-12
print('PASS: spin purity, exact fermionic 2-CSF Hamiltonian, 4 independent gradient/Hessian directions including CI response, normalized coherent overlap',flush=True)
