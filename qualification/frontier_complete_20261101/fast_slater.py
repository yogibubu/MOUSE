"""Scalar Schur Slater--Condon elements; no occupied four-index transform."""
from itertools import permutations
import numpy as np

def determinant(a):return float(np.linalg.det(a)) if len(a) else 1.
def cofactors(a):
    n=len(a);one=np.zeros_like(a);two=np.zeros((n,n,n,n))
    for i in range(n):
        for j in range(n):
            one[i,j]=(-1)**(i+j)*determinant(np.delete(np.delete(a,i,axis=0),j,axis=1))
            for k in range(n):
                if i==k:continue
                for l in range(n):
                    if j==l:continue
                    two[i,j,k,l]=(-1)**(i+j+k+l)*np.sign(k-i)*np.sign(l-j)*determinant(np.delete(np.delete(a,[i,k],axis=0),[j,l],axis=1))
    return one,two

def schur(a,b):
    u,s,vh=np.linalg.svd(a.T@b);phase=np.linalg.det(u)*np.linalg.det(vh);a=a@u;b=b@vh.T
    r=int(np.count_nonzero(s<.1));count=len(s)-r;m=a.T@b
    if r>4:raise ValueError('large null overlap block')
    if count:
        inv=np.linalg.inv(m[:count,:count]);det=phase*determinant(m[:count,:count]);aa=a[:,:count];bb=b[:,:count]
        regular=bb@inv@aa.T;left=b[:,count:]-bb@inv@m[:count,count:];right=a[:,count:]-aa@inv.T@m[count:,:count].T
        block=m[count:,count:]-m[count:,:count]@inv@m[:count,count:]
    else:det=phase;regular=np.zeros((len(a),len(a)));left=b;right=a;block=m
    first,second=cofactors(block);density=left@first.T@right.T
    return det,determinant(block),regular,density,left,right,second

class Hamiltonian:
    def __init__(self,h,eri):
        self.h=h;self.eri=eri;n=len(h);self.j=eri.reshape(n*n,n*n);self.k=eri.transpose(1,2,3,0).reshape(n*n,n*n);self.jk=self.j-self.k
    def coulomb(self,p,q):return float(p.ravel()@self.j@q.ravel())
    def same(self,p,q):return float(p.ravel()@self.jk@q.ravel())
    def element(self,bra,ket):
        p=[schur(a,b) for a,b in zip(bra,ket)];det=p[0][0]*p[1][0];total=0.
        for spin in range(2):
            _,d,r,q,l,v,second=p[spin];other=p[1-spin][1]
            one=d*np.trace(self.h@r)+np.trace(self.h@q)
            two=.5*d*self.same(r,r)+self.same(r,q)
            if l.shape[1]:
                pairs=np.einsum('pi,qj->ijpq',v,l).reshape(l.shape[1]**2,-1)
                tensor=(pairs@self.j@pairs.T).reshape(second.shape);two+=.5*float(np.sum(second*tensor))
            total+=other*(one+two)
        _,da,ra,qa,*_=p[0];_,db,rb,qb,*_=p[1]
        total+=da*db*self.coulomb(ra,rb)+da*self.coulomb(ra,qb)+db*self.coulomb(qa,rb)+self.coulomb(qa,qb)
        return det*da*db,det*total
    def csf(self,bra,ket):
        s=h=0.
        for x,a in bra:
            for y,b in ket:
                ov,value=self.element(a,b);s+=x*y*ov;h+=x*y*value
        return s,h
