"""Singular-safe determinant and adjugate mixed jets via a tiny null block."""
import numpy as np
from ._nosi_mixed import Jet,inverse,regular_determinant,small_determinant,cofactors

def determinant_adjugate(matrix,threshold=.1):
    u,s,vh=np.linalg.svd(matrix.v);v=vh.T;phase=np.linalg.det(u)*np.linalg.det(vh)
    m=matrix.coerce(u.T)@matrix@v;nullity=int(np.count_nonzero(s<threshold));count=len(s)-nullity
    if nullity>4:raise ValueError('determinant null block exceeds the qualified polynomial range')
    a=m[:count,:count];x=m[:count,count:];y=m[count:,:count]
    if count:
        inv=inverse(a);da=regular_determinant(a);schur=m[count:,count:]-y@inv@x
    else:
        inv=Jet.constant(np.zeros((0,0)),matrix.n);da=Jet.constant(1.,matrix.n);schur=m
    db=small_determinant(schur);adj=cofactors(schur)[0].T
    result=Jet.constant(np.zeros_like(m.v),matrix.n)
    result[:count,:count]=db*inv+inv@x@adj@y@inv
    result[:count,count:]=-inv@x@adj
    result[count:,:count]=-adj@y@inv
    result[count:,count:]=adj
    return phase*da*db,phase*da*(result.coerce(v)@result@u.T)

def ground_constraint(ground,excited,overlap,occupations,left,right):
    no=int(np.sum(occupations[0]));v=ground[:,:no].T@overlap@excited
    parts=[]
    for o in occupations:
        indices=np.flatnonzero(o);det,adj=determinant_adjugate(v[:,indices])
        full=Jet.constant(np.zeros_like(v.v),v.n);full[:,indices]=adj.T
        q=v.T@full;derivative=q[left,right]-q[right,left]
        parts.append((det,derivative))
    overlap_value=2**.5*parts[0][0]*parts[1][0]
    gradient=2**.5*(parts[1][0]*parts[0][1]+parts[0][0]*parts[1][1])
    return overlap_value,gradient
