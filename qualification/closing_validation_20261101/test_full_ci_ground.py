import numpy as np
from scipy.linalg import eigh
from matrix_gfn2._nosi_selected_roots import solve_full

def test_full_pencil_includes_ground_correlation_without_fitted_shift():
 h=np.array([[0.,0.,.2],[0.,1.,0.],[.2,0.,2.]])
 e,c,certificate=solve_full(h,np.eye(3),3)
 np.testing.assert_allclose(e,[1-np.sqrt(1.04),1.,1+np.sqrt(1.04)],atol=1e-13)
 assert abs((e[1]-e[0])-np.sqrt(1.04))<1e-13
 assert certificate['configurations_discarded']==0

def test_sparse_full_roots_are_generalized_and_energy_gauge_invariant():
 rng=np.random.default_rng(813);a=rng.normal(size=(12,12));s=a@a.T+np.eye(12);b=rng.normal(size=(12,12));h=(b+b.T)/2-40*s
 e,c,_=solve_full(h,s,4);expected=eigh(h,s,eigvals_only=True)[:4]
 np.testing.assert_allclose(e,expected,atol=1e-11)
 shifted,_,_=solve_full(h+17*s,s,4);np.testing.assert_allclose(shifted-e,17.,atol=1e-11)

def test_preconditioned_block_roots_match_dense_generalized_reference():
 rng=np.random.default_rng(851);n=70;a=rng.normal(size=(n,n))*.02;s=a@a.T+np.eye(n);b=rng.normal(size=(n,n))*.01;h=np.diag(np.linspace(.05,2,n))+(b+b.T)/2
 e,c,certificate=solve_full(h,s,4,energy_shift=0.,algorithm='lobpcg')
 np.testing.assert_allclose(e,eigh(h,s,eigvals_only=True)[:4],atol=1e-10)
 assert certificate['maximum_residual_hartree']<1e-8
