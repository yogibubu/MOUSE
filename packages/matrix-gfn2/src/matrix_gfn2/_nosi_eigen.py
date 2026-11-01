"""Analytic generalized-eigenvalue jets, including ground projection.

This algebra consumes complete matrix-element jets. It does not supply the
missing molecular orbital/geometry response and is not a nuclear-Hessian API.
Array convention: value (...), gradient (ncoord,...), Hessian (ncoord,ncoord,...).
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh

@dataclass(frozen=True)
class Jet:
    value: np.ndarray
    gradient: np.ndarray
    hessian: np.ndarray

def product(a,b):
    return Jet(a.value@b.value,
        np.einsum('aij,jk->aik',a.gradient,b.value)+np.einsum('ij,ajk->aik',a.value,b.gradient),
        np.einsum('abij,jk->abik',a.hessian,b.value)+np.einsum('ij,abjk->abik',a.value,b.hessian)
        +np.einsum('aij,bjk->abik',a.gradient,b.gradient)+np.einsum('bij,ajk->abik',a.gradient,b.gradient))

def transpose(a):return Jet(a.value.T,a.gradient.transpose(0,2,1),a.hessian.transpose(0,1,3,2))

def ground_projector(metric):
    s,ds,dds=metric.value,metric.gradient,metric.hessian;n=len(s);d=len(ds)
    if s[0,0]<=0:raise ValueError('nonpositive ground norm')
    a=s[0,1:]/s[0,0]
    da=(ds[:,0,1:]-ds[:,0,0,None]*a)/s[0,0]
    dda=(dds[:,:,0,1:]-dds[:,:,0,0,None]*a-ds[:,0,0,None,None]*da[None]-ds[None,:,0,0,None]*da[:,None])/s[0,0]
    p=np.eye(n)[:,1:];p[0]-=a;dp=np.zeros((d,n,n-1));dp[:,0]=-da
    ddp=np.zeros((d,d,n,n-1));ddp[:,:,0]=-dda
    return Jet(p,dp,ddp)

def project_matrices(hamiltonian,metric):
    p=ground_projector(metric)
    return product(product(transpose(p),hamiltonian),p),product(product(transpose(p),metric),p)

def eigenvalue_jet(hamiltonian,metric,root,*,minimum_gap=1e-8):
    h,s=hamiltonian.value,metric.value
    if np.min(np.linalg.eigvalsh(s))<1e-10:raise ValueError('unresolved metric linear dependence')
    e,c=eigh(h,s);root=int(root);energy=e[root];v=c[:,root]
    dh,ds=hamiltonian.gradient,metric.gradient
    d2h,d2s=hamiltonian.hessian,metric.hessian
    tangent=dh-energy*ds
    gradient=np.einsum('i,aij,j->a',v,tangent,v)
    norm_response=np.einsum('i,aij,j->a',v,ds,v)
    hessian=np.einsum('i,abij,j->ab',v,d2h-energy*d2s,v)
    hessian-=np.outer(gradient,norm_response)+np.outer(norm_response,gradient)
    for other in range(len(e)):
        if other==root:continue
        gap=energy-e[other]
        if abs(gap)<minimum_gap:raise ValueError('unresolved eigenvalue degeneracy')
        coupling=np.einsum('i,aij,j->a',c[:,other],tangent,v)
        hessian+=2*np.outer(coupling,coupling)/gap
    return Jet(np.array(energy),gradient,.5*(hessian+hessian.T))
