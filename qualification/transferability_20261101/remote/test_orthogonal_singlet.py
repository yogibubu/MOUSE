import socket
assert socket.getfqdn()=='oracle.sns.it'
import numpy as np
from scipy.linalg import expm
from orthogonal_singlet import determinant_jet,ground_overlap_jet,exponential_pullback,ground_overlap_hessian

def test_overlap_gradient_including_singular_determinants():
    rng=np.random.default_rng(17);n=6;no=3;ground=np.eye(n)
    occ=np.zeros((2,n));occ[:,:no]=1;occ[0,0]=0;occ[0,no]=1
    left,right=np.triu_indices(n,1)
    k=rng.normal(size=(n,n));k=.2*(k-k.T)
    for c in [np.eye(n),expm(k)]:
        value,gradient=ground_overlap_jet(ground,c,np.eye(n),occ,left,right)
        for index,(i,j) in enumerate(zip(left,right)):
            d=np.zeros((n,n));d[i,j]=1;d[j,i]=-1;eps=1e-5
            plus=ground_overlap_jet(ground,c@expm(eps*d),np.eye(n),occ,left,right)[0]
            minus=ground_overlap_jet(ground,c@expm(-eps*d),np.eye(n),occ,left,right)[0]
            assert abs((plus-minus)/(2*eps)-gradient[index])<2e-10

def test_overlap_global_exponential_gradient():
    rng=np.random.default_rng(19);n=6;no=3;ground=np.eye(n)
    occ=np.zeros((2,n));occ[:,:no]=1;occ[0,0]=0;occ[0,no]=1
    left,right=np.triu_indices(n,1);k=rng.normal(size=(n,n));k=.2*(k-k.T)
    rotation=expm(k);value,local=ground_overlap_jet(ground,rotation,np.eye(n),occ,left,right)
    gradient=exponential_pullback(k,rotation,left,right,local)
    for index,(i,j) in enumerate(zip(left,right)):
        d=np.zeros((n,n));d[i,j]=1;d[j,i]=-1;eps=1e-5
        plus=ground_overlap_jet(ground,expm(k+eps*d),np.eye(n),occ,left,right)[0]
        minus=ground_overlap_jet(ground,expm(k-eps*d),np.eye(n),occ,left,right)[0]
        assert abs((plus-minus)/(2*eps)-gradient[index])<2e-10

def test_analytic_overlap_curvature_at_singular_and_regular_points():
    rng=np.random.default_rng(23);n=6;no=3;ground=np.eye(n)
    occ=np.zeros((2,n));occ[:,:no]=1;occ[0,0]=0;occ[0,no]=1
    left,right=np.triu_indices(n,1);k=rng.normal(size=(n,n));k=.2*(k-k.T)
    for c in [np.eye(n),expm(k)]:
        h=ground_overlap_hessian(ground,c,np.eye(n),occ,left,right)
        numerical=np.zeros_like(h)
        for index,(i,j) in enumerate(zip(left,right)):
            d=np.zeros((n,n));d[i,j]=1;d[j,i]=-1;eps=1e-5
            plus=ground_overlap_jet(ground,c@expm(eps*d),np.eye(n),occ,left,right)[1]
            minus=ground_overlap_jet(ground,c@expm(-eps*d),np.eye(n),occ,left,right)[1]
            numerical[:,index]=(plus-minus)/(2*eps)
        numerical=.5*(numerical+numerical.T)
        assert np.max(abs(h-numerical))<2e-10
