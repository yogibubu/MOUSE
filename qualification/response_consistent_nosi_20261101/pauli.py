"""Representability projection allowing a general nonlocal two-body operator.

W_pqrs multiplies a^+_p a^+_r a_s a_q. General W has particle-exchange
W_pqrs=W_rspq and real Hermiticity W_pqrs=W_qpsr, but need not have the
extra independent p/q and r/s interchange of multiplicative Coulomb ERIs.
"""
import numpy as np

def pair_symmetric(w):
    return .25*(w+w.swapaxes(0,1)+w.swapaxes(2,3)+w.swapaxes(0,1).swapaxes(2,3))

def exchange(w):
    return pair_symmetric(w.transpose(0,3,1,2))

def fully_symmetric(v):
    return (v+2*exchange(v))/3

def project(charge,spin):
    incompatible=fully_symmetric(charge+spin)
    c=charge-.5*incompatible;s=spin-.5*incompatible
    v=c-s;d=-2*s-exchange(v);x=-(2/3)*d
    w=v+x.transpose(0,2,1,3)-x.transpose(0,2,3,1)
    return w,c,s,incompatible

def jk(w,p):
    return np.einsum('ijkl,kl->ij',w,p,optimize=True),np.einsum('iklj,kl->ij',w,p,optimize=True)

def fock(w,h,p):
    j,k=jk(w,p);return h+j-.5*k

def normal_ordered_one_body(w,target,p):
    return target-fock(w,np.zeros_like(target),p)
