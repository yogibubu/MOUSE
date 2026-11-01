"""Singular-safe Slater--Condon elements for equal alpha/beta counts.

Inputs occupy an orthonormal spatial AO basis. No inverse overlap is used;
products excluding one/two singular values include exactly orthogonal pairs.
"""
import numpy as np

def exclusion(s, count):
    if count == 1:
        return np.array([np.prod(np.delete(s,i)) for i in range(len(s))])
    return np.array([[0. if i==j else np.prod(np.delete(s,[i,j])) for j in range(len(s))] for i in range(len(s))])

def element(bra, ket, h, eri):
    a=[];b=[];singular=[];phase=1.
    for aa,bb in zip(bra,ket):
        u,s,vh=np.linalg.svd(aa.T@bb)
        a.append(aa@u);b.append(bb@vh.T);singular.append(s)
        phase*=np.linalg.det(u)*np.linalg.det(vh)
    determinants=[np.prod(s) for s in singular]
    one=[exclusion(s,1) for s in singular];two=[exclusion(s,2) for s in singular]
    overlap=phase*np.prod(determinants)
    value=0.
    for spin in range(2):
        value+=determinants[1-spin]*np.sum(one[spin]*np.einsum('pi,pq,qi->i',a[spin],h,b[spin]))
        v=np.einsum('pqrs,pi,qj,rk,sl->ijkl',eri,a[spin],b[spin],a[spin],b[spin],optimize=True)
        direct=np.einsum('iijj->ij',v);exchange=np.einsum('ijji->ij',v)
        value+=.5*determinants[1-spin]*np.sum(two[spin]*(direct-exchange))
    v=np.einsum('pqrs,pi,qi,rj,sj->ij',eri,a[0],b[0],a[1],b[1],optimize=True)
    value+=np.sum(one[0][:,None]*one[1][None,:]*v)
    return float(overlap),float(phase*value)

def csf_element(bra,ket,h,eri):
    ov=0.;value=0.
    for x,a in bra:
        for y,b in ket:
            s,e=element(a,b,h,eri);ov+=x*y*s;value+=x*y*e
    return ov,value
