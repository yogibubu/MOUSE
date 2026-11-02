import numpy as np
import pytest
from matrix_gfn2._nosi_static_subtraction import static_subtraction,subtracted_matrix_jet


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


def test_small_nonzero_response_column_is_solved_relatively():
    h=np.array([[.2,1e-16,2e-16],[1e-16,.4,.01],[2e-16,.01,.6]])
    r,delta,certificate=static_subtraction(h,np.eye(3),[0])
    np.testing.assert_allclose(delta,h[:1,1:]@np.linalg.solve(h[1:,1:],h[1:,:1]),atol=1e-44,rtol=1e-10)
    assert certificate['maximum_inverse_relative_residual']<1e-9


def test_complete_subtraction_jet_includes_ground_and_metric_hessian_response():
    from scipy.linalg import eigh
    from matrix_gfn2._mixed_jet import MixedJet as Jet
    from matrix_gfn2._nosi_nuclear import projected_eigen_jet
    rng=np.random.default_rng(73);n=7
    def symmetric(scale):
        a=rng.normal(size=(n,n))*scale;return (a+a.T)/2
    s0=np.eye(n)+symmetric(.02);s1=symmetric(.01);s2=symmetric(.01)
    a0=np.diag(np.linspace(0.,1.2,n))+symmetric(.01);a1=symmetric(.015);a2=symmetric(.01)
    for a in [a0,a1,a2]:a[0,0]=0.
    def coefficients(t):
        s=s0+t*s1+.5*t*t*s2;eg=-3+.05*t+.03*t*t
        return eg*s+a0+t*a1+.5*t*t*a2,s,eg
    def energy(t):
        h,s,eg=coefficients(t);p=np.eye(n)[:,1:];p[0]-=s[0,1:]/s[0,0]
        hp=p.T@(h-eg*s)@p;sp=p.T@s@p
        r,delta,_=static_subtraction(hp,sp,[0,2])
        return eigh(hp+r@delta@r.T,sp,eigvals_only=True)[0]+eg
    h0,_,_=coefficients(0.)
    h1=.05*s0-3*s1+a1;h2=.06*s0+.1*s1-3*s2+a2
    hj=Jet(h0,h1[None],h1,h2[None]);sj=Jet(s0,s1[None],s1,s2[None]);egj=Jet(-3.,np.array([.05]),.05,np.array([.06]))
    corrected=subtracted_matrix_jet(hj,sj,egj,[0,2]);answer=projected_eigen_jet(corrected,sj,0)
    step=2e-4;minus=energy(-step);zero=energy(0.);plus=energy(step)
    assert abs(float(answer.v)-zero)<1e-12
    assert abs(answer.a[0]-(plus-minus)/(2*step))<1e-8
    assert abs(answer.ab[0]-(plus-2*zero+minus)/step**2)<2e-7
