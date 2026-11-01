"""Exact common-core Schur acceleration versus direct nonorthogonal elements."""
from pathlib import Path
import socket,runpy,json,itertools,time
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
BASE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101');HERE=BASE/'oracle/MOUSE-closing-validation-20261101';runpy.run_path(str(BASE/'source/qualification/remote_bootstrap.py'))
import numpy as np
from scipy.linalg import expm
from matrix_gfn2._nosi_slater import Hamiltonian,cofactors
from matrix_gfn2._nosi_fixed_bra import FixedBra,FixedBraCSF,spectral_cofactors
from threadpoolctl import threadpool_limits

with threadpool_limits(2):
 rng=np.random.default_rng(382);rows=[]
 cofactor_error=0.
 for n in [2,4,5,6]:
  for rank in [n,n-1,max(1,n-2)]:
   matrix=rng.normal(size=(n,rank))@rng.normal(size=(rank,n));expected=cofactors(matrix);actual=spectral_cofactors(matrix);cofactor_error=max(cofactor_error,max(float(np.max(abs(a-b))) for a,b in zip(actual,expected)))
 assert cofactor_error<1e-8
 for n,no,nc,general in [(8,4,2,False),(12,6,2,False),(8,4,2,True),(12,6,2,True),(14,7,2,False),(16,8,2,False),(16,8,2,True)]:
  factors=rng.normal(size=(25,n,n));factors=.5*(factors+factors.transpose(0,2,1));eri=np.einsum('Lpq,Lrs->pqrs',factors,factors);h=rng.normal(size=(n,n));h=.5*(h+h.T);ham=Hamiltonian(h,eri);core=np.eye(n)[:,:nc];active=np.eye(n)[:,nc:];rotations=[]
  if general:
   eri=rng.normal(size=(n,)*4);eri=.5*(eri+eri.transpose(2,3,0,1));eri=.5*(eri+eri.transpose(1,0,3,2));ham=Hamiltonian(h,eri)
  for spin in range(2):
   k=.1*rng.normal(size=(n,n));rotations.append(expm(k-k.T)[:,:no])
  for same_spin in [False,True]:
   bra=tuple(rotations) if not same_spin else (rotations[0],rotations[0]);engine=FixedBra(ham,bra,core,active);error=0.
   occupations=[indices for indices in itertools.combinations(range(n-nc),no-nc) if sum(j>=no-nc for j in indices)<=2];ids=rng.integers(len(occupations),size=(12,2));start=time.monotonic()
   for ia,ib in ids:
    ket=tuple(np.column_stack([core,active[:,occupations[index]]]) for index in [ia,ib]);actual=engine.element(ket);expected=ham.element(bra,ket);error=max(error,float(np.max(abs(np.array(actual)-expected))))
   assert error<1e-9,(n,no,nc,error);rows.append(dict(orbitals=n,occupied_per_spin=no,common_core=nc,general_Hermitian_tensor=general,identical_alpha_beta=same_spin,maximum_overlap_or_energy_error=error,cases=len(ids),wall_seconds=time.monotonic()-start))
 report=dict(host=socket.getfqdn(),rows=rows,cofactor_max_error=cofactor_error,orbital_truncation=False,density_truncation=False,integral_truncation=False)
 (HERE/'fixed_bra_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
