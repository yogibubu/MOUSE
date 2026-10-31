"""Independent scalar differences verify analytic transport and relaxed response."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.linalg import expm
from matrix_gfn2.nonlocal_response import rotation_jet,TransportedOperators,PairTerm


def setup():
    rng=np.random.default_rng(85);xyz=np.array([[0.,0.,0.],[1.3,.1,0.],[-.4,1.1,0.]])
    state=SimpleNamespace(atomic_numbers=np.array([6,1,1]),coordinates_bohr=xyz,
        overlap=np.eye(6),ao_atom=np.array([0,0,0,0,1,2]),ao_shell=['2s','2p','2p','2p','1s','1s'])
    a=rng.normal(size=(7,6,6));a=(a+a.transpose(0,2,1))/2
    k=rng.normal(size=(7,7));k=(k+k.T)*.02
    p=rng.normal(size=(6,6));p=(p+p.T)/2
    return rng,state,a,k,p


def test_rotation_jet_planar_and_singular_frame():
    rng,s,a,k,p=setup();xyz=s.coordinates_bohr.copy();xyz[1,2]+=.07
    r,dr,d2r=rotation_jet(xyz,s.coordinates_bohr)
    d=rng.normal(size=xyz.shape);t=2e-4
    rp=rotation_jet(xyz+t*d,s.coordinates_bohr)[0];rm=rotation_jet(xyz-t*d,s.coordinates_bohr)[0]
    assert np.max(abs((rp-rm)/(2*t)-np.einsum('r,rij->ij',d.ravel(),dr)))<2e-7
    assert np.max(abs((rp+rm-2*r)/t**2-np.einsum('r,s,rsij->ij',d.ravel(),d.ravel(),d2r)))<5e-7
    with pytest.raises(ValueError,match='singular'):rotation_jet(np.array([[0.,0.,0.],[1.,0.,0.]]),np.array([[0.,0.,0.],[1.,0.,0.]]))


def test_nonlocal_geometry_fock_and_energy():
    rng,s,a,k,p=setup();anchor=s.coordinates_bohr.copy();s.coordinates_bohr=anchor+.03*rng.normal(size=anchor.shape)
    ds=rng.normal(size=(9,6,6))*.01;ds=(ds+ds.transpose(0,2,1))/2
    d2s=rng.normal(size=(9,9,6,6))*.01;d2s=(d2s+d2s.swapaxes(0,1))/2;d2s=(d2s+d2s.swapaxes(2,3))/2
    q0=rng.normal(size=7);v=rng.normal(size=7)*.1
    def term(x):
        state=SimpleNamespace(**vars(s));state.coordinates_bohr=s.coordinates_bohr+x.reshape(-1,3)
        state.overlap=s.overlap+np.einsum('r,rij->ij',x,ds)+.5*np.einsum('r,t,rtij->ij',x,x,d2s)
        return PairTerm(TransportedOperators(state,anchor,a),k,q0,v)
    zero=np.zeros(9);base=term(zero);df,h=base.geometry(p,ds,d2s)
    analytic=np.einsum('ij,rij->r',base.metric_potential(p),ds)+base.explicit_gradient(p).ravel()
    direction=rng.normal(size=9);t=1e-4;plus=term(t*direction);minus=term(-t*direction)
    assert (plus.evaluate(p)[1]-minus.evaluate(p)[1])/(2*t)==pytest.approx(analytic@direction,abs=3e-7)
    assert (plus.evaluate(p)[1]+minus.evaluate(p)[1]-2*base.evaluate(p)[1])/t**2==pytest.approx(direction@h@direction,abs=2e-6)
    tf=1e-5;fp=term(tf*direction);fm=term(-tf*direction)
    assert np.max(abs((fp.evaluate(p)[0]-fm.evaluate(p)[0])/(2*tf)-np.einsum('r,rij->ij',direction,df)))<2e-8
    # Rotating coordinates, AO metric and density leaves the SAME model energy invariant.
    g=rng.normal(size=(3,3));rot=expm(.4*(g-g.T));u=np.eye(6);u[1:4,1:4]=rot
    rs=SimpleNamespace(**vars(s));rs.coordinates_bohr=s.coordinates_bohr@rot.T+np.array([1.,2.,-3.]);rs.overlap=u@s.overlap@u.T
    rt=PairTerm(TransportedOperators(rs,anchor,a),k,q0,v)
    assert rt.evaluate(u@p@u.T)[1]==pytest.approx(base.evaluate(p)[1],abs=1e-11)


def test_batched_response_matches_scalar_operator_and_orbital_jacobian():
    from matrix_gfn2.nonlocal_response import PairSpin
    from matrix_gfn2.spin_polarization import ShellSpinPolarization
    from matrix_gfn2.roks_orbital_response import orbital_residual_jacobian
    rng,s,a,k,p=setup()
    transport=TransportedOperators(s,s.coordinates_bohr.copy(),a)
    term=PairTerm(transport,k)
    spin=PairSpin(ShellSpinPolarization(s.overlap,np.arange(6),.03*np.eye(6)),PairTerm(transport,.2*k))
    dp=rng.normal(size=(11,2,6,6));dp=.5*(dp+dp.swapaxes(-1,-2))
    np.testing.assert_allclose(term.response_batch(dp.sum(axis=1)),
                              [term.response(x.sum(axis=0)) for x in dp],atol=2e-12)
    np.testing.assert_allclose(spin.response_batch(dp),
                              [spin.evaluate(x)[0] for x in dp],atol=2e-12)
    c=np.eye(6);occ=np.array([[1,1,0,0,0,0],[1,0,1,0,0,0]])
    f=rng.normal(size=(2,6,6));f=.5*(f+f.swapaxes(-1,-2))
    left,right=np.triu_indices(6,1)
    def kernel(x):return term.response(x)
    kernel.response_batch=term.response_batch
    fast=orbital_residual_jacobian(c,occ,f,kernel,spin,left,right)
    scalar=type('ScalarSpin',(),{'evaluate':lambda self,p:spin.evaluate(p)})()
    slow=orbital_residual_jacobian(c,occ,f,lambda p:term.response(p),scalar,left,right)
    np.testing.assert_allclose(fast,slow,atol=2e-12)


def test_relaxed_nonlocal_hessian():
    from matrix_gfn2.nonlocal_response import PairSpin
    from matrix_gfn2.spin_polarization import ShellSpinPolarization
    from matrix_gfn2.roks import run_roks_singlet
    from matrix_gfn2.deltascf_analytic_hessian import _response_blocks
    rng,s,a,k,p=setup();anchor=s.coordinates_bohr.copy();direction=rng.normal(size=anchor.shape)*.1
    ds=rng.normal(size=(6,6))*.01;ds=(ds+ds.T)/2;d2s=.01*np.eye(6)
    h0=np.diag([-1.,-.7,-.4,.2,.5,.8]);h1=np.diag(np.arange(6))*.003;occ=np.array([[1,1,0,0,0,0],[1,0,1,0,0,0]])
    def build(x):
        ss=SimpleNamespace(**vars(s));ss.coordinates_bohr=anchor+x*direction;ss.overlap=np.eye(6)+x*ds+.5*x*x*d2s
        tr=TransportedOperators(ss,anchor,a)
        spin=PairSpin(ShellSpinPolarization(ss.overlap,np.arange(6),np.zeros((6,6))),PairTerm(tr,k*.003))
        charge=PairTerm(tr,k*.004,v=np.arange(7)*.0001)
        def fun(p):
            sf,se,_=spin.evaluate(p);cf,ce=charge.evaluate(p.sum(axis=0));h=h0+x*h1
            return (h+cf)[None]+sf,np.sum(h*p.sum(axis=0))+se+ce
        return ss,spin,charge,fun
    ss,spin,charge,fun=build(0.)
    r=run_roks_singlet(ss.overlap,np.eye(6),occ,fun,orbital_tolerance=1e-11)
    # Contract the full 9-coordinate operator jets into this scalar nuclear path.
    def contract(term):
        tr=term.transport;first=tr.first().copy();second=np.array([[tr.second(i,j) for j in range(9)] for i in range(9)])
        tr._first=np.einsum('r,raij->aij',direction.ravel(),first)[None]
        d2a=np.einsum('r,s,rsaij->aij',direction.ravel(),direction.ravel(),second)
        tr.second=lambda i,j:d2a
    contract(spin.term) # charge shares this exact transport instance
    c=r.common_coefficients;components=[(2.,occ,r.fock_mixed),(-1.,r.triplet_occupations,r.fock_triplet)];explicit=[]
    for _,o,_ in components:
        total=sum((c*z)@c.T for z in o);df,rr=charge.geometry(total,ds[None],d2s[None,None]);explicit.append((h1[None]+df,rr))
    rr,aa,b=_response_blocks([c],components,ds[None],d2s[None,None],explicit,[charge.response]*2,spin)
    analytic=float((rr-b.T@np.linalg.solve(aa,b))[0,0]);t=2e-4;values=[]
    for x in [-t,0,t]:
        ss,_,_,f=build(x);ev,q=np.linalg.eigh(c.T@ss.overlap@c)
        values.append(run_roks_singlet(ss.overlap,c@((q*ev**-.5)@q.T),occ,f,orbital_tolerance=1e-11).energy_hartree)
    assert analytic==pytest.approx((values[0]+values[2]-2*values[1])/t**2,abs=2e-6)


def test_nonlocal_reference_serialization_and_fingerprint(tmp_path):
    from dataclasses import replace
    from matrix_gfn2 import FrozenNonlocalResponseReference,save_response_reference,load_response_reference
    rng,s,a,k,p=setup();partition=SimpleNamespace(shell_atom=np.arange(4),orbital_shell=np.array([0,1,1,1,2,3]))
    data=dict(operators=a,charge=k,spin=k*.3,q0=np.zeros(7),linear=np.arange(7)*.001)
    ref=FrozenNonlocalResponseReference.from_vertical_model(s,partition,data,'synthetic independent response')
    path=tmp_path/'response.npz';save_response_reference(path,ref)
    loaded=load_response_reference(path);assert loaded.fingerprint==ref.fingerprint
    assert replace(ref,pair_linear_hartree=ref.pair_linear_hartree+.01).fingerprint!=ref.fingerprint
    assert not loaded.operators.flags.writeable
    with np.load(path) as d:payload={k:d[k] for k in d.files}
    payload['pair_charge_hartree']*=.9;np.savez_compressed(path,**payload)
    with pytest.raises(ValueError,match='checksum'):load_response_reference(path)


def test_composite_keeps_two_stationary_states_and_relaxed_hessians(monkeypatch):
    from dataclasses import replace
    import matrix_gfn2.composite_roks as module
    from matrix_gfn2.roks import run_roks_singlet
    h=np.diag([-.7,-.1]);occ=np.array([[1.,0.],[0.,1.]])
    raw=run_roks_singlet(np.eye(2),np.eye(2),occ,lambda p:(np.array([h,h]),float(np.sum(h*p))))
    state=SimpleNamespace(atomic_numbers=np.array([2]),coordinates_bohr=np.zeros((1,3)),
        ao_shell=['1s','2s'],ao_atom=[0,0],ao_exponents=np.array([1.,2.]),charge=0,multiplicity=1,model_id='gfn2-test')
    raw=replace(raw,reference_state=state,core_electrons=np.array([0]),ground_energy_hartree=-1.4,
        gradient_hartree_per_bohr=np.array([[.1,.2,.3]]),spin_reference_fingerprint='raw')
    spectral=replace(raw,energy_hartree=raw.energy_hartree+.2,ground_energy_hartree=-1.3,
        gradient_hartree_per_bohr=np.array([[-.3,.4,.1]]),spin_reference_fingerprint='spectral')
    result=module.combine_roks_results(raw,spectral)
    assert result.component_csf_overlap==pytest.approx(1)
    assert result.excitation_energy_hartree==pytest.approx(.5*(raw.excitation_energy_hartree+spectral.excitation_energy_hartree))
    assert result.gradient_hartree_per_bohr==pytest.approx(np.array([[-.1,.3,.2]]))
    calls=[]
    def derivative(calc,xyz,*occupations,**kw):
        calls.append(kw['reference_result']);return SimpleNamespace(hessian_hartree_per_bohr2=(2 if calc=='raw' else 4)*np.eye(3))
    monkeypatch.setattr(module,'evaluate_gfn2_deltascf_analytic_hessian',derivative)
    hessian=module.evaluate_composite_roks_analytic_hessian(('raw','spectral'),np.zeros((1,3)),('r','s'),reference_result=result)
    assert hessian.hessian_hartree_per_bohr2==pytest.approx(3*np.eye(3))
    assert calls==[raw,spectral]
    with pytest.raises(ValueError,match='stationary'):module.combine_roks_results(raw,replace(spectral,orbital_residual=1e-4))
    with pytest.raises(ValueError,match='geometries'):
        other=SimpleNamespace(**vars(state));other.coordinates_bohr=np.ones((1,3))
        module.combine_roks_results(raw,replace(spectral,reference_state=other))
