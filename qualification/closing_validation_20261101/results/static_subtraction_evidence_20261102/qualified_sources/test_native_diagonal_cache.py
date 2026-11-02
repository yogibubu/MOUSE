from types import SimpleNamespace
import numpy as np
from matrix_gfn2._nosi_surface_response import MatrixResponse
from matrix_gfn2._nosi_configurations import from_labels


def test_native_functional_cache_survives_screened_operator_change():
    n=6;rng=np.random.default_rng(235);y=np.eye(n);q=np.linalg.qr(rng.normal(size=(n,n)))[0]
    records=from_labels(['single_h1_p4','pair_h1_p3','open_h1_1_p3_4','open_h1_2_p3_3','four_h1_2_p3_4_0','four_h1_2_p3_4_1'],3)
    ctx=SimpleNamespace(state=SimpleNamespace(electron_count=6,overlap=np.eye(n),coordinates_bohr=np.zeros((2,3))),charge_correction=SimpleNamespace(transport=SimpleNamespace(u=np.eye(n))))
    ctx.energy=lambda p:float(-np.trace(p.sum(0))+.05*np.sum(p.sum(0)**2))
    ref=SimpleNamespace(operators=np.empty((0,n,n)),pair_spin_hartree=np.empty((0,0)))
    a=rng.normal(size=(n,n));h=(a+a.T)/2;w=rng.normal(size=(n,n,n,n));w=(w+w.transpose(2,3,0,1))/2;w=(w+w.transpose(1,0,3,2))/2
    def build(h,w,**kwargs):
        obj=MatrixResponse(ctx,ref,y,[q],[0.],y,['unused']*(3+len(records)),h,w,derivatives=False,auxiliary_records=records,**kwargs)
        return obj.matrices(np.zeros(6))
    old,metric,eg=build(h,w);values={i:float(old.v[i,i]/metric.v[i,i]) for i in range(len(old.v))};four={7:old.v[7:9,7:9].copy()}
    fresh,sfresh,_=build(h+.02*np.eye(n),w*.8)
    ctx.energy=lambda *args:(_ for _ in ()).throw(AssertionError('completed native functional value recomputed'))
    cached,scached,ecached=build(h+.02*np.eye(n),w*.8,native_diagonals=values,native_four_blocks=four)
    np.testing.assert_allclose(cached.v,fresh.v,atol=1e-11)
    np.testing.assert_allclose(scached.v,sfresh.v,atol=1e-12)
    assert abs(float(ecached.v-eg.v))<1e-12
