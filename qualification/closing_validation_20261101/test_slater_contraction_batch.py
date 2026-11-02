"""Exact batching must preserve nonsingular and zero-overlap Slater elements."""
import numpy as np
from matrix_gfn2._nosi_slater import Hamiltonian,schur

def test_batched_elements_preserve_regular_and_null_overlap_sectors():
 rng=np.random.default_rng(481);n=7;occ=3
 h=rng.normal(size=(n,n));h=(h+h.T)/2
 factors=rng.normal(size=(9,n,n));factors=(factors+factors.transpose(0,2,1))/2
 w=np.einsum('Lpq,Lrs->pqrs',factors,factors)
 canonical=np.eye(n);rotated=np.linalg.qr(rng.normal(size=(n,n)))[0]
 ground=(canonical[:,:occ],canonical[:,:occ])
 cases=[(ground,(rotated[:,:occ],rotated[:,:occ])),(ground,(canonical[:,[0,1,3]],canonical[:,:occ])),(ground,(canonical[:,[0,3,4]],canonical[:,[0,1,3]]))]
 reference=Hamiltonian(h,w);expected=[reference.element(a,b) for a,b in cases]
 accelerated=Hamiltonian(h,w);densities=[]
 for a,b in cases:
  for x,y in zip(a,b):densities.extend(schur(x,y)[2:4])
 accelerated.prefetch_csf_pairs([([(1.,a)],[(1.,b)]) for a,b in cases])
 actual=[accelerated.element(a,b) for a,b in cases]
 np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
 # Reversed bra/ket symmetry provides a separate physical consistency check.
 np.testing.assert_allclose([accelerated.element(b,a) for a,b in cases],expected,atol=1e-12,rtol=1e-12)
