"""Irreducible tensor covariance and full analytic relaxed Hessian checks."""
from types import SimpleNamespace
from dataclasses import replace,asdict
import numpy as np
import pytest
from matrix_gfn2 import FrozenTensorResponseReference,save_response_reference,load_response_reference
from matrix_gfn2.tensor_spin import onsite_spin_generators
from matrix_gfn2.roks import run_roks_singlet
from matrix_gfn2.deltascf_analytic_hessian import _response_blocks


def bind(s):
    state=SimpleNamespace(atomic_numbers=np.array([6]),ao_atom=np.zeros(4,dtype=int),
        ao_shell=['2s','2p','2p','2p'],overlap=s)
    partition=SimpleNamespace(orbital_shell=np.array([0,1,1,1]))
    r=FrozenTensorResponseReference(np.array([6]),partition.orbital_shell,
        np.array([[-.1,-.02],[-.02,-.09]]),'test',.9,.04*np.eye(2),np.array([1.8,2.2]),
        np.array([[-.03,-.02]]),np.array([[.04,.03]]))
    return r,r.bind(state,partition),r.bind_charge(state,partition)


def test_tensor_covariance_variational_fock_and_model_roundtrip(tmp_path):
    rng=np.random.default_rng(8);x=rng.normal(size=(4,4));s=np.eye(4)+.04*(x+x.T)
    r,spin,charge=bind(s);p=rng.normal(size=(2,4,4));p=(p+p.transpose(0,2,1))/2
    dp=rng.normal(size=p.shape);dp=(dp+dp.transpose(0,2,1))/2;step=1e-5
    f,e,_=spin.evaluate(p);fc,ec=charge.evaluate(p.sum(axis=0));f=f+fc
    def energy(q):return spin.evaluate(q)[1]+charge.evaluate(q.sum(axis=0))[1]
    assert (energy(p+step*dp)-energy(p-step*dp))/(2*step)==pytest.approx(np.sum(f*dp),abs=1e-9)
    rot=np.eye(4);rot[1:,1:]=np.linalg.qr(rng.normal(size=(3,3)))[0]
    _,sr,cr=bind(rot.T@s@rot);pr=rot.T@p@rot
    fr,er,_=sr.evaluate(pr);fcr,ecr=cr.evaluate(pr.sum(axis=0))
    assert er+ecr==pytest.approx(e+ec,abs=1e-12)
    assert fr+fcr==pytest.approx(rot.T@f@rot,abs=1e-12)
    file=tmp_path/'response.npz';save_response_reference(file,r);assert load_response_reference(file).fingerprint==r.fingerprint
    assert replace(r,onsite_charge_hartree=r.onsite_charge_hartree*2).fingerprint!=r.fingerprint
    with np.load(file) as d:data={k:d[k] for k in d.files}
    data['onsite_spin_hartree']*=.9;np.savez_compressed(file,**data)
    with pytest.raises(ValueError,match='checksum'):load_response_reference(file)


def test_tensor_charge_and_spin_relaxed_analytic_hessian():
    rng=np.random.default_rng(20)
    def sym(scale):
        a=rng.normal(size=(4,4))*scale;return (a+a.T)/2
    s1=sym(.06);s2=sym(.03);h0=np.diag([-.8,-.4,-.17,.45])+sym(.01);h1=sym(.04);h2=sym(.03)
    occ=np.array([[1,1,0,0],[1,0,1,0]])
    def setup(x):
        s=np.eye(4)+x*s1+.5*x*x*s2;_,spin,charge=bind(s);h=h0+x*h1+.5*x*x*h2
        def callback(p):
            total=p.sum(axis=0);sf,se,_=spin.evaluate(p);cf,ce=charge.evaluate(total)
            return (h+.07*total+cf)[None]+sf,np.sum(h*total)+.035*np.sum(total*total)+se+ce
        return s,spin,charge,callback
    s,spin,charge,callback=setup(0.);r=run_roks_singlet(s,np.eye(4),occ,callback,orbital_tolerance=1e-11)
    c=r.common_coefficients;components=[(2.,occ,r.fock_mixed),(-1.,r.triplet_occupations,r.fock_triplet)];explicit=[]
    for _,o,_ in components:
        p=sum((c*x)@c.T for x in o);df,rr=charge.geometry(p,s1[None],s2[None,None])
        explicit.append((h1[None]+df,np.array([[np.sum(p*h2)]])+rr))
    rr,a,b=_response_blocks([c],components,s1[None],s2[None,None],explicit,[lambda p:.07*p+charge.response(p)]*2,spin)
    analytic=float((rr-b.T@np.linalg.solve(a,b))[0,0]);values=[];step=2e-4
    for x in [-step,0,step]:
        ss,_,_,fun=setup(x);v,q=np.linalg.eigh(c.T@ss@c)
        values.append(run_roks_singlet(ss,c@((q*v**-.5)@q.T),occ,fun,orbital_tolerance=1e-11).energy_hartree)
    assert analytic==pytest.approx((values[0]-2*values[1]+values[2])/step**2,abs=5e-7)


def test_link_no_offset_response_roundtrip_and_dispatch(tmp_path,monkeypatch):
    from matrix_link.scan import QMScanBackend,_composite_restart_json_value
    from matrix_link.deltascf_component import evaluate_component
    import matrix_gfn2
    r,_,_=bind(np.eye(4));path=tmp_path/'response.npz';save_response_reference(path,r)
    b=QMScanBackend(name='matrix_gfn2',electronic_state=1,native_excited_method='roks',
        native_roks_response_path=path,deltascf_promotion=(1,2),orbital_saddle_index=1)
    assert QMScanBackend(**asdict(b)).native_roks_response_sha256==b.native_roks_response_sha256
    assert 'native_roks_response_path' not in _composite_restart_json_value(QMScanBackend(name='matrix_gfn2'))
    result=SimpleNamespace(energy_hartree=-1.,gradient_hartree_per_bohr=np.zeros((1,3)),spin_squared=0.,
        orbital_residual=0.,orbital_saddle_index=1,to_apoc=lambda i:None)
    calls=[]
    def energy(*a,**kw):calls.append(kw['spin_reference'].fingerprint);return result
    def hessian(*a,**kw):calls.append(kw['spin_reference'].fingerprint);return SimpleNamespace(hessian_hartree_per_bohr2=np.eye(3))
    monkeypatch.setattr(matrix_gfn2,'evaluate_gfn2_roks_singlet',energy)
    monkeypatch.setattr(matrix_gfn2,'evaluate_gfn2_deltascf_analytic_hessian',hessian)
    c=SimpleNamespace(backend=b,calculator=SimpleNamespace(_calculator=None),reference=None,anchor=None)
    ground=SimpleNamespace(orbital_state=SimpleNamespace(alpha_occupations=np.array([1,1,0,0]),beta_occupations=np.array([1,1,0,0])))
    point=evaluate_component(c,np.zeros((1,3)),True,ground,{})
    assert calls==[r.fingerprint,r.fingerprint] and point.execution['no_offset_response_fingerprint']==r.fingerprint
    with path.open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='changed during'):evaluate_component(c,np.zeros((1,3)),False,ground,{})


def test_first_order_fock_term_is_varied_and_differentiated(tmp_path):
    from matrix_gfn2 import FrozenFockResponseReference
    state=SimpleNamespace(atomic_numbers=np.array([6]),ao_atom=np.zeros(4,dtype=int),ao_shell=['2s','2p','2p','2p'],overlap=np.eye(4))
    partition=SimpleNamespace(orbital_shell=np.array([0,1,1,1]));ref,_,_=bind(state.overlap)
    f=FrozenFockResponseReference(ref.atomic_numbers,ref.orbital_shell,ref.coupling_hartree,'test-fock',.9,
        ref.charge_correction_hartree,ref.reference_shell_populations,np.array([.1,-.02]))
    charge=f.bind_charge(state,partition);rng=np.random.default_rng(3)
    def sym():
        a=rng.normal(size=(4,4));return (a+a.T)/2
    p=sym();dp=sym();ds=sym()*.02;d2s=sym()*.01;step=1e-5
    field,energy=charge.evaluate(p)
    assert (charge.evaluate(p+step*dp)[1]-charge.evaluate(p-step*dp)[1])/(2*step)==pytest.approx(np.sum(field*dp),abs=1e-9)
    analytic_f,analytic_rr=charge.geometry(p,ds[None],d2s[None,None])
    points=[]
    for x in [-step,0,step]:
        sh=SimpleNamespace(**vars(state))
        sh.overlap=np.eye(4)+x*ds+.5*x*x*d2s;points.append(f.bind_charge(sh,partition).evaluate(p))
    assert analytic_f[0]==pytest.approx((points[2][0]-points[0][0])/(2*step),abs=1e-9)
    assert analytic_rr[0,0]==pytest.approx((points[2][1]-2*points[1][1]+points[0][1])/step**2,abs=1e-5)
    assert np.sum(charge.metric_potential(p)*ds)==pytest.approx((points[2][1]-points[0][1])/(2*step),abs=1e-9)
    path=tmp_path/'fock.npz';save_response_reference(path,f);assert load_response_reference(path).fingerprint==f.fingerprint
