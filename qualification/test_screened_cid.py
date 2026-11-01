"""Spin recoupling and exact canonical contraction for screened OO-CID."""
import numpy as np
import pytest
from matrix_gfn2._nosi_canonical import block
from matrix_gfn2._nosi_configurations import from_labels,validate,basis
from matrix_gfn2.state_interaction import Hamiltonian
from matrix_apoc import NonOrthogonalSingletCIState


def records():
    return from_labels(['pair_h1_p3','open_h1_1_p3_4','open_h1_2_p3_3',
                        'four_h1_2_p3_4_0','four_h1_2_p3_4_1'],3)


def test_coherent_extended_singlet():
    r=validate(records(),3,6);b=basis(np.eye(6),r);weights=np.arange(1,len(b)+1,dtype=float);weights/=np.linalg.norm(weights)
    determinants=[];coefficients=[]
    for weight,csf in zip(weights,b):
        for c,pair in csf:determinants.append(pair);coefficients.append(weight*c)
    state=NonOrthogonalSingletCIState(1,tuple(determinants),np.array(coefficients),np.eye(6))
    assert state.spin_squared==0.


@pytest.mark.parametrize('nonlocal_operator',[False,True])
def test_canonical_contraction_matches_nonorthogonal_algebra(nonlocal_operator):
    rng=np.random.default_rng(192);y=np.linalg.qr(rng.normal(size=(6,6)))[0];h=rng.normal(size=(6,6));h=(h+h.T)/2;w=rng.normal(size=(6,)*4)
    if nonlocal_operator:
        w=(w+w.transpose(2,3,0,1))/2;w=(w+w.transpose(1,0,3,2))/2
    else:
        w=(w+w.swapaxes(0,1))/2;w=(w+w.swapaxes(2,3))/2;w=(w+w.transpose(2,3,0,1))/2
    b=basis(y,records());metric,matrix,_=block(b,y,h,w,[0],[1,2,3,4]);literal=Hamiltonian(h,w)
    for i,a in enumerate(b):
        for j,c in enumerate(b):
            ov,value=literal.csf(a,c)
            assert abs(metric[i,j]-ov)<2e-12
            assert abs(matrix[i,j]-value)<2e-11


def test_rejects_incomplete_or_unphysical_recoupling():
    r=records()
    with pytest.raises(ValueError,match='four-spin'):validate(r[:-1],3,6)
    with pytest.raises(ValueError,match='indices'):validate([dict(kind='open',closed=[0,1],open=[2,2])],3,6)
    with pytest.raises(ValueError,match='occupation'):validate([dict(kind='closed',closed=[0,1])],3,6)
