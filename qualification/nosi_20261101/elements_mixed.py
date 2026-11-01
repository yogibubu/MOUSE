"""Analytic singular-safe Slater--Condon mixed jets in a fixed AO frame.

Schur-complement factorization isolates at most four near-null occupied
overlaps. Determinants/cofactors in that tiny block are division-free
polynomials. Only the well-conditioned regular block is inverted. No inverse
of the singular full occupied overlap and no derivative steps are used.
This is an element-jet building block, not a complete nuclear-response API.
"""
from itertools import permutations
import numpy as np
from matrix_gfn2._mixed_jet import MixedJet as Jet,contract,stack

def inverse(a):
    v=np.linalg.inv(a.v);first=-v@a.a@v;tangent=-v@a.b@v
    mixed=v@a.a@v@a.b@v+v@a.b@v@a.a@v-v@a.ab@v
    return Jet(v,first,tangent,mixed)

def regular_determinant(a):
    if not len(a.v):return Jet.constant(1.,a.n)
    inv=inverse(a);value=float(np.linalg.det(a.v))
    first=np.einsum('ij,aji->a',inv.v,a.a);tangent=np.trace(inv.v@a.b)
    mixed=np.einsum('ij,aji->a',inv.v,a.ab)-np.einsum('ij,ajk,kl,li->a',inv.v,a.a,inv.v,a.b)
    return Jet(value,value*first,value*tangent,value*(mixed+first*tangent))

def small_determinant(a):
    n=len(a.v)
    if n==0:return Jet.constant(1.,a.n)
    total=Jet.constant(0.,a.n)
    for p in permutations(range(n)):
        sign=(-1)**sum(p[i]>p[j] for i in range(n) for j in range(i+1,n));value=Jet.constant(float(sign),a.n)
        for i,j in enumerate(p):value=value*a[i,j]
        total=total+value
    return total

def minor(a,rows,cols):
    return a[np.ix_(np.delete(np.arange(len(a.v)),rows),np.delete(np.arange(len(a.v)),cols))]

def cofactors(a):
    n=len(a.v);one=Jet.constant(np.zeros((n,n)),a.n);two=Jet.constant(np.zeros((n,n,n,n)),a.n)
    for i in range(n):
        for j in range(n):
            one[i,j]=(-1)**(i+j)*small_determinant(minor(a,[i],[j]))
            for k in range(n):
                if k==i:continue
                for l in range(n):
                    if l==j:continue
                    sign=(-1)**(i+j+k+l)*np.sign(k-i)*np.sign(l-j)
                    two[i,j,k,l]=sign*small_determinant(minor(a,[i,k],[j,l]))
    return one,two

def rotated(a,u):
    return Jet(a.v@u,a.a@u,a.b@u,a.ab@u)

def schur(bra,ket,threshold=.1):
    u,s,vh=np.linalg.svd(bra.v.T@ket.v);phase=np.linalg.det(u)*np.linalg.det(vh)
    nullity=int(np.count_nonzero(s<threshold));count=len(s)-nullity
    if nullity>4:raise ValueError('occupied overlap requires a larger polynomial null block')
    a=rotated(bra,u);b=rotated(ket,vh.T);m=contract('pi,pj->ij',a,b)
    aa=a[:,:count];bb=b[:,:count];block=m[:count,:count]
    if count:
        inv=inverse(block);det=regular_determinant(block)
        regular=contract('pi,ij,qj->pq',bb,inv,aa)
        left=b[:,count:]-contract('pi,ij,jk->pk',bb,inv,m[:count,count:])
        right=a[:,count:]-contract('pi,ji,kj->pk',aa,inv,m[count:,:count])
        small=m[count:,count:]-contract('ij,jk,kl->il',m[count:,:count],inv,m[:count,count:])
    else:
        det=Jet.constant(1.,bra.n);regular=Jet.constant(np.zeros((len(a.v),len(a.v))),bra.n)
        left=b;right=a;small=m
    first,second=cofactors(small);small_det=small_determinant(small)
    null_density=contract('pj,ij,qi->pq',left,first,right)
    return phase*det,small_det,regular,null_density,left,right,second

def bilinear(w,x,y,exchange=False):
    expression='pqrs,sp,qr->' if exchange else 'pqrs,qp,sr->'
    def value(a,b):return np.einsum(expression,w,a,b,optimize=True)
    # Constants carry no large derivative arrays (the AO ERI tensor is fixed).
    first_expression='pqrs,asp,qr->a' if exchange else 'pqrs,aqp,sr->a'
    second_expression='pqrs,sp,aqr->a' if exchange else 'pqrs,qp,asr->a'
    one=lambda a,b:np.einsum(first_expression,w,a,b,optimize=True)
    two=lambda a,b:np.einsum(second_expression,w,a,b,optimize=True)
    return Jet(value(x.v,y.v),one(x.a,y.v)+two(x.v,y.a),value(x.b,y.v)+value(x.v,y.b),
        one(x.ab,y.v)+two(x.v,y.ab)+one(x.a,y.b)+two(x.b,y.a))

def trace(h,p):
    return Jet(np.einsum('pq,qp->',h,p.v),np.einsum('pq,aqp->a',h,p.a),np.einsum('pq,qp->',h,p.b),np.einsum('pq,aqp->a',h,p.ab))

def null_pair(w,left,right,second):
    n=left.v.shape[1]
    if not n:return Jet.constant(0.,left.n)
    pairs=contract('pi,qj->ijpq',right,left)
    def contraction(a,b):return np.einsum('pqrs,ijpq,klrs->ijkl',w,a,b,optimize=True)
    val=contraction(pairs.v,pairs.v)
    first=np.einsum('pqrs,aijpq,klrs->aijkl',w,pairs.a,pairs.v,optimize=True)+np.einsum('pqrs,ijpq,aklrs->aijkl',w,pairs.v,pairs.a,optimize=True)
    tangent=contraction(pairs.b,pairs.v)+contraction(pairs.v,pairs.b)
    mixed=np.einsum('pqrs,aijpq,klrs->aijkl',w,pairs.ab,pairs.v,optimize=True)+np.einsum('pqrs,ijpq,aklrs->aijkl',w,pairs.v,pairs.ab,optimize=True)
    mixed+=np.einsum('pqrs,aijpq,klrs->aijkl',w,pairs.a,pairs.b,optimize=True)+np.einsum('pqrs,ijpq,aklrs->aijkl',w,pairs.b,pairs.a,optimize=True)
    return .5*contract('ijkl,ijkl->',second,Jet(val,first,tangent,mixed))

def element(bra,ket,h,w):
    pieces=[schur(a,b) for a,b in zip(bra,ket)]
    determinant=[p[0] for p in pieces];d=[p[1] for p in pieces];r=[p[2] for p in pieces];q=[p[3] for p in pieces]
    total=Jet.constant(0.,bra[0].n)
    coulomb=lambda x,y:bilinear(w,x,y)
    same=lambda x,y:coulomb(x,y)-bilinear(w,x,y,True)
    for spin in range(2):
        p=pieces[spin]
        one=d[spin]*trace(h,r[spin])+trace(h,q[spin])
        two=.5*d[spin]*same(r[spin],r[spin])+same(r[spin],q[spin])+null_pair(w,p[4],p[5],p[6])
        total=total+d[1-spin]*(one+two)
    total=total+d[0]*d[1]*coulomb(r[0],r[1])+d[0]*coulomb(r[0],q[1])+d[1]*coulomb(q[0],r[1])+coulomb(q[0],q[1])
    factor=determinant[0]*determinant[1]
    return factor*d[0]*d[1],factor*total
