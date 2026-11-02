import numpy as np
import pytest
from matrix_gfn2._nosi_static_subtraction import static_subtraction


def test_nonorthogonal_subtraction_preserves_primary_static_response():
    rng=np.random.default_rng(62);a=rng.normal(size=(9,9));s=a@a.T+np.eye(9)
    b=rng.normal(size=(9,9));h=b@b.T+2*s;p=np.array([1,4,7]);q=np.setdiff1d(np.arange(9),p)
    r,delta,certificate=static_subtraction(h,s,p)
    x=np.linalg.solve(s[np.ix_(p,p)],s[np.ix_(p,q)])
    t=np.eye(9);t[np.ix_(p,q)]=-x
    original=t.T@h@t;corrected=t.T@(h+r@delta@r.T)@t
    v=corrected[np.ix_(p,q)];d=corrected[np.ix_(q,q)]
    np.testing.assert_allclose(corrected[np.ix_(p,p)]-v@np.linalg.solve(d,v.T),original[np.ix_(p,p)],atol=2e-11)
    np.testing.assert_allclose(corrected[np.ix_(q,q)],original[np.ix_(q,q)],atol=2e-11)
    assert np.linalg.eigvalsh(delta)[0]>0
    assert certificate['energy_offset_ev']==0


def test_subtraction_has_no_effect_without_primary_complement_coupling():
    h=np.diag([.1,.3,.5]);r,delta,_=static_subtraction(h,np.eye(3),[0])
    np.testing.assert_array_equal(delta,np.zeros((1,1)))


def test_unstable_complement_is_rejected():
    with pytest.raises(ValueError,match='positive complementary'):
        static_subtraction(np.diag([1.,-1.,2.]),np.eye(3),[0])
