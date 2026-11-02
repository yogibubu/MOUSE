"""Exact determinant overlap and local orbital gradient for a single singlet."""
import numpy as np
from scipy.linalg import expm_frechet

def exponential_pullback(k,rotation,left,right,local):
    z=np.zeros_like(k);z[left,right]=.5*local;z[right,left]=-.5*local
    derivative=expm_frechet(k.T,rotation@z,compute_expm=False)
    return derivative[left,right]-derivative[right,left]

def determinant_jet(a):
    u,s,vh=np.linalg.svd(a)
    phase=np.linalg.det(u)*np.linalg.det(vh)
    products=np.array([np.prod(np.delete(s,i)) for i in range(len(s))])
    return float(np.linalg.det(a)),phase*(u*products)@vh

def ground_overlap_jet(ground,c,overlap,occupations,left,right):
    no=int(np.sum(occupations[0]));ov=ground[:,:no].T@overlap@c
    indices=[np.flatnonzero(o) for o in occupations]
    da,ga=determinant_jet(ov[:,indices[0]]);db,gb=determinant_jet(ov[:,indices[1]])
    derivative=np.zeros_like(ov)
    derivative[:,indices[0]]+=np.sqrt(2)*db*ga
    derivative[:,indices[1]]+=np.sqrt(2)*da*gb
    q=ov.T@derivative
    return float(np.sqrt(2)*da*db),q[left,right]-q[right,left]


def ground_overlap_action(ground,c,overlap,occupations,left,right,vector,*,symmetric=True):
    """Singular-safe analytic overlap-Hessian action, with O(n^2) storage."""
    return prepare_ground_overlap_action(ground,c,overlap,occupations,left,right,symmetric=symmetric)(vector)


def prepare_ground_overlap_action(ground,c,overlap,occupations,left,right,*,symmetric=True):
    """Factor the fixed overlap once for all Krylov vectors at one geometry."""
    no=int(np.sum(occupations[0]));ov=ground[:,:no].T@overlap@c
    def determinant(indices):
        a=ov[:,indices];u,s,vh=np.linalg.svd(a)
        phase=np.linalg.det(u)*np.linalg.det(vh)
        products=np.array([np.prod(np.delete(s,i)) for i in range(no)])
        cofactor=phase*(u*products)@vh
        weights=np.zeros((no,no))
        for i in range(no):
            for j in range(no):
                if i!=j:weights[i,j]=phase*np.prod(np.delete(s,[i,j]))
        return float(np.linalg.det(a)),cofactor,u,vh,weights
    ia,ib=[np.flatnonzero(o) for o in occupations]
    first,second=determinant(ia),determinant(ib);da,ga=first[:2];db,gb=second[:2]
    derivative=np.zeros_like(ov)
    derivative[:,ia]+=np.sqrt(2)*db*ga;derivative[:,ib]+=np.sqrt(2)*da*gb
    q=ov.T@derivative
    def action(vector):
        rotation=np.zeros_like(c);rotation[left,right]=vector;rotation[right,left]=-vector;change=ov@rotation
        def variation(data,indices):
            value,cofactor,u,vh,weights=data;d=change[:,indices];x=u.T@d@vh.T;dcofactor=-weights*x.T
            np.fill_diagonal(dcofactor,weights@np.diag(x))
            return float(np.sum(cofactor*d)),u@dcofactor@vh
        dda,dga=variation(first,ia);ddb,dgb=variation(second,ib);mixed=np.zeros_like(ov)
        mixed[:,ia]+=np.sqrt(2)*(ddb*ga+db*dga);mixed[:,ib]+=np.sqrt(2)*(dda*gb+da*dgb)
        dq=change.T@derivative+ov.T@mixed
        if symmetric:dq+=.5*(rotation@q-q@rotation)
        return float(np.sqrt(2)*(dda*db+da*ddb)),dq[left,right]-dq[right,left]
    return action

def ground_overlap_hessian(ground,c,overlap,occupations,left,right):
    """Analytic exponential-chart curvature, including singular determinants."""
    no=int(np.sum(occupations[0]));n=len(c);d=len(left)
    ov=ground[:,:no].T@overlap@c
    generators=np.zeros((d,n,n));generators[np.arange(d),left,right]=1
    generators[np.arange(d),right,left]=-1
    variations=np.einsum('ij,ajk->aik',ov,generators,optimize=True)
    def determinant(indices):
        a=ov[:,indices];value,cofactor=determinant_jet(a)
        u,s,vh=np.linalg.svd(a);phase=np.linalg.det(u)*np.linalg.det(vh)
        x=u.T@variations[:,:,indices]@vh.T
        weights=np.zeros((no,no))
        for i in range(no):
            for j in range(no):
                if i!=j:weights[i,j]=phase*np.prod(np.delete(s,[i,j]))
        diagonal=np.diagonal(x,axis1=1,axis2=2)
        h=diagonal@weights@diagonal.T-np.einsum('ij,aij,bji->ab',weights,x,x,optimize=True)
        euclidean=np.zeros_like(ov);euclidean[:,indices]=cofactor;q=ov.T@euclidean
        g=np.einsum('ij,aij->a',q,generators)
        moving=(generators.transpose(0,2,1)@q).reshape(d,-1)@generators.reshape(d,-1).T
        return value,g,h+.5*(moving+moving.T)
    da,ga,ha=determinant(np.flatnonzero(occupations[0]))
    db,gb,hb=determinant(np.flatnonzero(occupations[1]))
    h=np.sqrt(2)*(db*ha+da*hb+np.outer(ga,gb)+np.outer(gb,ga))
    return .5*(h+h.T)
