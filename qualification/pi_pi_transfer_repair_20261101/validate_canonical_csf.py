"""Compare canonical bitstring contraction to the independent nonorthogonal code."""
from pathlib import Path
import sys,json,socket,time
assert socket.getfqdn()=='oracle.sns.it'
ROOT=Path('/home/enzo/MOUSE-response-consistent-nosi-20261101');HERE=Path('/home/enzo/MOUSE-pi-pi-transfer-repair-20261101')
sys.path.insert(0,str(ROOT))
from basis import *
from canonical_csf import block
from itertools import combinations
with threadpool_limits():
 rng=np.random.default_rng(814);errors=[]
 for n,no,ncore in [(6,3,1),(7,4,2)]:
  y=np.linalg.qr(rng.normal(size=(n,n)))[0];h=rng.normal(size=(n,n));h=.5*(h+h.T);raw=rng.normal(size=(n,n,n,n));w=.5*(raw+raw.swapaxes(0,1));w=.5*(w+w.swapaxes(2,3));w=.5*(w+w.transpose(2,3,0,1));ham=Hamiltonian(h,w);core=list(range(ncore));active=list(range(ncore,n));basis=[]
  # Determinants of both spins and arbitrary signed linear combinations.
  dets=[]
  for a in combinations(active,no-ncore):
   for b in combinations(active,no-ncore):dets.append((y[:,core+list(a)],y[:,core+list(b)]))
  for _ in range(24):
   terms=[]
   for idx in rng.choice(len(dets),size=3,replace=False):
    aa,bb=dets[idx];pa=rng.permutation(no);pb=rng.permutation(no);terms.append((float(rng.normal()),(aa[:,pa],bb[:,pb])))
   basis.append(terms)
  start=time.monotonic();s,a,info=block(basis,y,h,w,core,active);fast=time.monotonic()-start;ss=np.zeros_like(s);bb=np.zeros_like(a);start=time.monotonic()
  for i in range(len(basis)):
   for j in range(i+1):ov,value=ham.csf(basis[i],basis[j]);ss[i,j]=ss[j,i]=ov;bb[i,j]=bb[j,i]=value
  error=dict(n=n,metric_error=float(np.max(abs(s-ss))),energy_matrix_error=float(np.max(abs(a-bb))),fast_seconds=fast,independent_seconds=time.monotonic()-start,**info);assert error['metric_error']<1e-10 and error['energy_matrix_error']<1e-9,error;errors.append(error)
 out=HERE/'canonical_csf_validation.json';out.write_text(json.dumps(errors,indent=2)+'\n');print(errors)
