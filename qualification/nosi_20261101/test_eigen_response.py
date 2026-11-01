import socket
assert socket.getfqdn()=='oracle.sns.it'
import numpy as np
from scipy.linalg import eigh
from eigen_response import *

def test_complete_projected_response_against_energy_differences():
    rng=np.random.default_rng(29);n=5;d=2
    a0=np.eye(n)+.1*rng.normal(size=(n,n));a1=.1*rng.normal(size=(d,n,n));a2=.1*rng.normal(size=(d,d,n,n));a2=.5*(a2+a2.swapaxes(0,1))
    h0=np.diag(np.arange(n)*.7)+.1*rng.normal(size=(n,n));h0=.5*(h0+h0.T)
    h1=.1*rng.normal(size=(d,n,n));h1=.5*(h1+h1.transpose(0,2,1))
    h2=.1*rng.normal(size=(d,d,n,n));h2=.5*(h2+h2.transpose(0,1,3,2));h2=.5*(h2+h2.swapaxes(0,1))
    s0=a0.T@a0;s1=np.array([x.T@a0+a0.T@x for x in a1]);s2=np.empty_like(a2)
    for i in range(d):
        for j in range(d):s2[i,j]=a2[i,j].T@a0+a0.T@a2[i,j]+a1[i].T@a1[j]+a1[j].T@a1[i]
    hj,sj=project_matrices(Jet(h0,h1,h2),Jet(s0,s1,s2));result=eigenvalue_jet(hj,sj,1)
    def energy(x):
        a=a0+np.einsum('a,aij->ij',x,a1)+.5*np.einsum('a,b,abij->ij',x,x,a2)
        s=a.T@a;h=h0+np.einsum('a,aij->ij',x,h1)+.5*np.einsum('a,b,abij->ij',x,x,h2)
        p=np.eye(n)[:,1:];p[0]-=s[0,1:]/s[0,0]
        return eigh(p.T@h@p,p.T@s@p,eigvals_only=True)[1]
    eps=1e-4;eye=np.eye(d);zero=np.zeros(d);base=energy(zero)
    gradient=np.array([(energy(eps*x)-energy(-eps*x))/(2*eps) for x in eye])
    hh=np.empty((d,d))
    for i in range(d):
        for j in range(d):
            if i==j:hh[i,j]=(energy(eps*eye[i])-2*base+energy(-eps*eye[i]))/eps**2
            else:hh[i,j]=(energy(eps*(eye[i]+eye[j]))-energy(eps*(eye[i]-eye[j]))-energy(eps*(-eye[i]+eye[j]))+energy(-eps*(eye[i]+eye[j])))/(4*eps**2)
    assert np.max(abs(gradient-result.gradient))<1e-8
    assert np.max(abs(hh-result.hessian))<5e-7

def test_zero_overlap_and_degeneracy_guard():
    s=Jet(np.eye(3),np.zeros((1,3,3)),np.zeros((1,1,3,3)));h=Jet(np.diag([0.,1.,1.]),s.gradient,s.hessian)
    hp,sp=project_matrices(h,s)
    import pytest
    with pytest.raises(ValueError,match='degeneracy'):eigenvalue_jet(hp,sp,0)
