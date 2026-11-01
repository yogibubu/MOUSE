"""Spin recoupling and exact canonical contraction for screened OO-CID."""
import numpy as np
import pytest
from matrix_gfn2._nosi_canonical import block
from matrix_gfn2._nosi_configurations import from_labels,validate,basis
from matrix_gfn2.state_interaction import Hamiltonian
from matrix_apoc import NonOrthogonalSingletCIState


def records():
    return from_labels(['single_h1_p4','pair_h1_p3','open_h1_1_p3_4','open_h1_2_p3_3',
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


@pytest.mark.parametrize('label',['single_h3_p4','single_h1_p2'])
def test_rejects_nonexciting_single(label):
    with pytest.raises(ValueError,match='single-excitation'):from_labels([label],3)


@pytest.mark.parametrize('dimension',[0,3])
def test_zero_aware_mixed_contractions_preserve_product_rule(dimension):
    from matrix_gfn2._mixed_jet import MixedJet,contract
    rng=np.random.default_rng(46);shape=(3,3)
    for frozen in [False,True]:
        x=MixedJet(rng.normal(size=shape),rng.normal(size=(dimension,*shape)),rng.normal(size=shape),rng.normal(size=(dimension,*shape)))
        y=MixedJet.constant(rng.normal(size=shape),dimension) if frozen else MixedJet(rng.normal(size=shape),rng.normal(size=(dimension,*shape)),rng.normal(size=shape),rng.normal(size=(dimension,*shape)))
        actual=contract('ij,jk->ik',x,y)
        assert np.allclose(actual.v,x.v@y.v,atol=2e-14,rtol=0)
        assert np.allclose(actual.b,x.b@y.v+x.v@y.b,atol=2e-14,rtol=0)
        for coordinate in range(dimension):
            assert np.allclose(actual.a[coordinate],x.a[coordinate]@y.v+x.v@y.a[coordinate],atol=2e-14,rtol=0)
            expected=x.ab[coordinate]@y.v+x.v@y.ab[coordinate]+x.a[coordinate]@y.b+x.b@y.a[coordinate]
            assert np.allclose(actual.ab[coordinate],expected,atol=2e-14,rtol=0)


def test_cached_numpy_program_preserves_greedy_arithmetic():
    from matrix_gfn2._einsum_cached import einsum
    rng=np.random.default_rng(113)
    examples=[('ij,jk->ik',[(3,4),(4,2)]),('pqrs,sp,qr->',[(3,)*4,(3,3),(3,3)]),
              ('pqrs,asp,qr->a',[(3,)*4,(2,3,3),(3,3)]),
              ('...ij,...jk->...ik',[(2,3,4),(2,4,5)]),('i,ij,j->',[(3,),(3,3),(3,)])]
    for expression,shapes in examples:
        operands=[rng.normal(size=shape) for shape in shapes]
        expected=np.einsum(expression,*operands,optimize=True)
        for _ in range(2):assert np.array_equal(einsum(expression,*operands,optimize=True),expected)


def test_frozen_auxiliary_metric_reuse_is_exact():
    from matrix_gfn2.nosi_surface import _cross_metric
    from scipy.linalg import expm
    rng=np.random.default_rng(12);k=rng.normal(size=(6,6));rotation=expm(.1*(k-k.T));aux=basis(np.eye(6),records())
    left=[[(1.,(np.eye(6)[:,:3],np.eye(6)[:,:3]))]]+aux
    right=[[(1.,(rotation[:,:3],rotation[:,:3]))]]+aux
    full=_cross_metric(left,right);gram=_cross_metric(aux,aux)
    assert np.array_equal(full,_cross_metric(left,right,nbase=1,auxiliary_gram=gram))
