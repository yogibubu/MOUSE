"""Parameter-free closest fermionic Hamiltonian to a charge/spin kernel pair.

All tensors use an orthonormal spatial basis and symmetric density directions.
T(V)[ij,kl] = (V[ik,jl] + V[il,jk])/2. On eightfold ERI-symmetric
tensors T*T=(I+T)/2. The least-squares objective is
||V-T(V)/2-C||_F^2 + ||-T(V)/2-S||_F^2.
Its unique minimizer is (11 C - 3 S + 2 T(C) - 10 T(S))/13.
No excitation-energy datum, weight or offset is fitted.
"""
import numpy as np

def exchange(v):
    return .5*(v.transpose(0,2,1,3)+v.transpose(0,2,3,1))

def hamiltonian_projection(charge,spin):
    return (11*charge-3*spin+2*exchange(charge)-10*exchange(spin))/13

def response_residuals(v,charge,spin):
    t=exchange(v)
    return v-.5*t-charge,-.5*t-spin

def fock(v,h,p):
    return h+np.einsum('ijkl,kl->ij',v,p,optimize=True)-.5*np.einsum('ikjl,kl->ij',v,p,optimize=True)

def normal_ordered_one_body(v,target_fock,reference_density):
    return target_fock-fock(v,np.zeros_like(target_fock),reference_density)
