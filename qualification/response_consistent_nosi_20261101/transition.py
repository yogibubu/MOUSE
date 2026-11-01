"""Exact one-body CSF transition densities, including singular overlaps."""
import numpy as np
from matrix_gfn2._nosi_slater import schur

def transition_density(bra,ket):
    pieces=[schur(a,b) for a,b in zip(bra,ket)]
    p=np.zeros((len(bra[0]),len(bra[0])))
    for spin in range(2):
        det,d,r,q,*_=pieces[spin];other=pieces[1-spin]
        p+=det*(d*r+q)*(other[0]*other[1])
    return p

def csf_transition(ground,csf):
    return sum(weight*transition_density(ground,pair) for weight,pair in csf)

def vectors_in_frame(cg,basis,no):
    ground=(cg[:,:no],cg[:,:no]);cv=cg[:,no:];co=cg[:,:no]
    return np.array([(co.T@csf_transition(ground,b).T@cv).ravel()/np.sqrt(2) for b in basis])

def csf_one_body(bra,ket,operator):
    value=0.
    for wa,a in bra:
        for wb,b in ket:value+=wa*wb*np.trace(operator@transition_density(a,b))
    return float(value)
