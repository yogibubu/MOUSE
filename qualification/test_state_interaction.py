"""Portable public state-interaction regression, no electronic calculations."""
import numpy as np
from scipy.linalg import eigh
from matrix_gfn2.state_interaction import (
    Hamiltonian, MatrixDerivatives, eigenvalue_jet,
    four_open_shell_singlets, four_open_shell_spin_matrix,
)


def test_generalized_eigen_hessian_includes_metric_response():
    rng=np.random.default_rng(128);n=4;d=2
    h=rng.normal(size=(n,n));h=h+h.T;s=np.eye(n)+.03*np.ones((n,n))
    dh=rng.normal(size=(d,n,n));dh=.5*(dh+dh.transpose(0,2,1));ds=.03*dh
    hh=rng.normal(size=(d,d,n,n));hh=.25*(hh+hh.transpose(1,0,2,3)+hh.transpose(0,1,3,2)+hh.transpose(1,0,3,2));ss=.01*hh
    jet=eigenvalue_jet(MatrixDerivatives(h,dh,hh),MatrixDerivatives(s,ds,ss),1)
    direction=np.array([.6,-.8]);step=2e-4
    def energy(t):
        hp=h+t*np.einsum('a,aij->ij',direction,dh)+.5*t*t*np.einsum('a,b,abij->ij',direction,direction,hh)
        sp=s+t*np.einsum('a,aij->ij',direction,ds)+.5*t*t*np.einsum('a,b,abij->ij',direction,direction,ss)
        return eigh(hp,sp,eigvals_only=True)[1]
    assert abs((energy(step)-energy(-step))/(2*step)-direction@jet.gradient)<1e-6
    assert abs((energy(step)-2*energy(0)+energy(-step))/step**2-direction@jet.hessian@direction)<2e-6


def test_four_spin_singlets_and_isotropic_exchange():
    basis=four_open_shell_singlets(np.eye(4),2);ham=Hamiltonian(np.zeros((4,4)),np.zeros((4,)*4))
    overlap=np.array([[ham.csf(a,b)[0] for b in basis] for a in basis])
    np.testing.assert_allclose(overlap,np.eye(2),atol=1e-12)
    pair=np.full((4,4),.2);np.fill_diagonal(pair,.3)
    np.testing.assert_allclose(four_open_shell_spin_matrix(1.,pair),np.eye(2)*.4,atol=1e-12)


from matrix_gfn2._mixed_jet import MixedJet as Jet
from matrix_gfn2.state_interaction import determinant_element_mixed

def test_singular_safe_element_jets():
    rng=np.random.default_rng(53);n=6;occupied=3;d=2
    h=rng.normal(size=(n,n));h=.5*(h+h.T);v=rng.normal(size=(n,n,n,n));v=.5*(v+v.swapaxes(0,1));v=.5*(v+v.swapaxes(2,3));v=.5*(v+v.transpose(2,3,0,1))
    for nullity in [0,1,2,3]:
        a=np.eye(n)[:,:occupied];b=np.eye(n)[:,list(range(nullity,occupied))+list(range(occupied,occupied+nullity))]
        bases=[a,a,b,b];first=[.1*rng.normal(size=(d,n,occupied)) for _ in bases];second=[.1*rng.normal(size=(d,d,n,occupied)) for _ in bases]
        second=[.5*(x+x.swapaxes(0,1)) for x in second]
        jets=[Jet(x,dx,dx[1],ddx[:,1]) for x,dx,ddx in zip(bases,first,second)]
        analytic=determinant_element_mixed(jets[:2],jets[2:],h,v)
        def curve(x):
            c=[base+np.einsum('a,aij->ij',x,da)+.5*np.einsum('a,b,abij->ij',x,x,dda) for base,da,dda in zip(bases,first,second)]
            return np.array(Hamiltonian(h,v).element(c[:2],c[2:]))
        zero=curve(np.zeros(d));assert np.max(abs(zero-[x.v for x in analytic]))<1e-10
        eps=2e-4;eye=np.eye(d)
        for axis in range(d):
            gradient=(curve(eps*eye[axis])-curve(-eps*eye[axis]))/(2*eps)
            assert np.max(abs(gradient-[x.a[axis] for x in analytic]))<1e-7
            if axis==1:mixed=(curve(eps*eye[axis])-2*zero+curve(-eps*eye[axis]))/eps**2
            else:mixed=(curve(eps*(eye[axis]+eye[1]))-curve(eps*(eye[axis]-eye[1]))-curve(eps*(-eye[axis]+eye[1]))+curve(-eps*(eye[axis]+eye[1])))/(4*eps**2)
            assert np.max(abs(mixed-[x.ab[axis] for x in analytic]))<2e-6
