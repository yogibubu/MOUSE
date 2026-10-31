import numpy as np
import pytest
from matrix_apoc import (DeterminantOrbitals,DeltaSCFState,SpinAdaptedSinglet,
    SingletCSFExpansion,multicsf_overlap,follow_multicsf_state)

def csfs():
 result=[]
 for i in range(2):
  occ=np.zeros((2,4));occ[:,:2]=1;occ[0,i]=0;occ[0,2+i]=1
  wf=DeterminantOrbitals(np.array([2,2]),np.zeros((2,3)),np.eye(4),(np.eye(4),np.eye(4)),tuple(occ),np.zeros(2,dtype=int),model='synthetic')
  result.append(SpinAdaptedSinglet(1,DeltaSCFState(1,wf)))
 return tuple(result)

def test_coherent_expansion_interference_and_phase():
 c=csfs();plus=SingletCSFExpansion(1,c,np.array([1.,1.])/np.sqrt(2));minus=SingletCSFExpansion(2,c,np.array([1.,-1.])/np.sqrt(2));flipped=SingletCSFExpansion(3,c,-plus.coefficients)
 assert multicsf_overlap(plus,minus,np.eye(4))[0]<1e-12
 score,phase=multicsf_overlap(plus,flipped,np.eye(4));assert abs(score-1)<1e-12 and phase==-1
 chosen,match=follow_multicsf_state(plus,[minus,flipped],np.eye(4));assert chosen.state_id==3 and match.phase==-1

def test_expansion_rejects_duplicate_csfs_and_wrong_normalization():
 c=csfs()
 with pytest.raises(ValueError,match='orthonormal'):SingletCSFExpansion(1,(c[0],c[0]),np.array([1.,1.])/np.sqrt(2))
 with pytest.raises(ValueError,match='normalized'):SingletCSFExpansion(1,c,np.array([1.,1.]))
