"""Reassemble a prepared vertical CI pencil with a different screened interaction.

The anchored native functional, orbitals and configurations must be identical.
Only the off-diagonal screened Hamiltonian and its diagonal normalization change;
all native diagonal and four-spin functional values are reused exactly.
"""
import json,time
import numpy as np
from ._nosi_configurations import common_core_and_active
from ._nosi_canonical import block
from ._nosi_slater import Hamiltonian
from ._nosi_fixed_bra import FixedBraCSF


def reassemble(y,basis,records,nbase,h,w,parent_h,parent_s,*,elements,checkpoint,coulomb_factors=None):
    core,active=common_core_and_active(records)
    cs,ch,details=block(None,y,h,w,core,active,records=records)
    error=float(np.max(abs(cs-parent_s[nbase:,nbase:])))
    if error>1e-10:raise ValueError(('rescreening changed canonical metric',error))
    scalar=Hamiltonian(h,w);scalar.coulomb_factors=coulomb_factors;fixed=FixedBraCSF(scalar,y[:,core],y[:,active]);raw=np.zeros_like(parent_h);raw[nbase:,nbase:]=ch
    scalar.prefetch_csf_pairs((basis[i],basis[j]) for i in range(nbase)
                             for j in range(i+1) if (i,j) not in elements)
    fixed.prefill(basis,nbase,elements,checkpoint)
    for i in range(len(basis)):
        for j in range(min(i+1,nbase)):
            if (i,j) not in elements:
                elements[i,j]=fixed.element(basis[j],basis[i]) if i>=nbase else scalar.csf(basis[i],basis[j])
                checkpoint(elements)
            ov,value=elements[i,j]
            error=max(error,abs(ov-parent_s[i,j]))
            if abs(ov-parent_s[i,j])>1e-10:raise ValueError(('rescreening changed orbital identity',i,j,ov,float(parent_s[i,j])))
            raw[i,j]=raw[j,i]=value
    native=np.diag(parent_h)/np.diag(parent_s)
    residual=native-np.diag(raw)/np.diag(parent_s)
    result=raw+.5*parent_s*(residual[:,None]+residual[None,:])
    for i,rec in enumerate(records):
        if rec['kind']=='four' and rec['component']==0:
            j=nbase+i;result[j:j+2,j:j+2]=parent_h[j:j+2,j:j+2]
    checkpoint(elements,force=True)
    return result,parent_s.copy(),dict(native_functional_values_reused=True,screened_operator_recomputed=True,maximum_metric_identity_error=error,canonical_block=details)
