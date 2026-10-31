"""Derivative and identity tests for a frozen, variational charge/spin kernel."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from matrix_gfn2.spin_reference import (FrozenSpinReference,FrozenResponseReference,
                                      ShellChargeCorrection,validate_spin_reference)
from matrix_gfn2.spin_polarization import ShellSpinPolarization
from matrix_gfn2.roks import run_roks_singlet
from matrix_gfn2.deltascf_analytic_hessian import _response_blocks


def test_reference_identity_covers_charge_origin_kernel_and_lambda():
    r=FrozenResponseReference(np.array([1,1]),np.array([0,1]),-.1*np.eye(2),
        'test',.9,.03*np.eye(2),np.ones(2))
    assert len({r.fingerprint,replace(r,response_exchange_scale=.8).fingerprint,
                replace(r,reference_shell_populations=np.array([1.1,.9])).fingerprint,
                replace(r,charge_correction_hartree=.04*np.eye(2)).fingerprint})==4
    with pytest.raises(ValueError):r.charge_correction_hartree[0,0]=0
    validate_spin_reference(SimpleNamespace(spin_reference_fingerprint=r.fingerprint),r)
    with pytest.raises(ValueError,match='functional differ'):
        validate_spin_reference(SimpleNamespace(spin_reference_fingerprint=r.fingerprint),None)
    with pytest.raises(ValueError):
        replace(r,charge_correction_hartree=[[1,2],[0,1]])


def test_charge_energy_fock_and_kernel_are_the_same_variational_functional():
    s=np.array([[1,.14],[.14,1]])
    u=np.array([[.12,-.03],[-.03,-.04]])
    charge=ShellChargeCorrection(s,np.array([0,1]),u,np.array([.9,1.1]))
    p=np.array([[1.3,.08],[.08,.7]]);dp=np.array([[.17,.09],[.09,-.13]])
    f,_=charge.evaluate(p);step=1e-5
    fp,ep=charge.evaluate(p+step*dp);fm,em=charge.evaluate(p-step*dp)
    assert (ep-em)/(2*step)==pytest.approx(np.sum(f*dp),abs=1e-10)
    assert (fp-fm)/(2*step)==pytest.approx(charge.response(dp),abs=1e-10)


def test_relaxed_hessian_includes_charge_metric_and_shared_orbital_response():
    s1=np.array([[.10,.04,-.03],[.04,-.03,.02],[-.03,.02,.07]])
    s2=np.array([[.02,-.01,.03],[-.01,.01,.02],[.03,.02,-.02]])
    h0=np.array([[-.8,.02,.03],[.02,-.4,-.01],[.03,-.01,-.17]])
    h1=np.array([[.06,.02,-.01],[.02,-.03,.03],[-.01,.03,.01]])
    h2=np.array([[.03,.01,.02],[.01,.04,-.01],[.02,-.01,-.02]])
    w=np.array([[-.10,-.02,-.01],[-.02,-.13,-.03],[-.01,-.03,-.09]])
    u=np.array([[.06,-.02,.01],[-.02,.09,.02],[.01,.02,-.03]])
    occ=np.array([[1,1,0],[1,0,1]])
    def setup(r):
        s=np.eye(3)+r*s1+.5*r*r*s2
        spin=ShellSpinPolarization(s,np.arange(3),w)
        charge=ShellChargeCorrection(s,np.arange(3),u,np.array([1.8,1.1,1.1]))
        h=h0+r*h1+.5*r*r*h2
        def callback(p):
            total=p.sum(axis=0);sf,se,_=spin.evaluate(p);cf,ce=charge.evaluate(total)
            f=h+.07*total+cf
            return f[None]+sf,np.sum(h*total)+.035*np.sum(total*total)+se+ce
        return s,spin,charge,callback
    s,spin,charge,callback=setup(0.)
    result=run_roks_singlet(s,np.eye(3),occ,callback,orbital_tolerance=1e-11)
    c=result.common_coefficients;components=[(2.,occ,result.fock_mixed),(-1.,result.triplet_occupations,result.fock_triplet)]
    explicit=[]
    for _,o,_ in components:
        p=sum((c*x)@c.T for x in o)
        df,rr=charge.geometry(p,s1[None],s2[None,None])
        explicit.append((h1[None]+df,np.array([[np.sum(p*h2)]])+rr))
    rr,a,b=_response_blocks([c],components,s1[None],s2[None,None],explicit,
        [lambda dp:.07*dp+charge.response(dp)]*2,spin)
    analytic=(rr-b.T@np.linalg.solve(a,b))[0,0]
    values=[];step=2e-4
    for r in [-step,0,step]:
        ss,_,_,fun=setup(r);v,q=np.linalg.eigh(c.T@ss@c)
        trial=run_roks_singlet(ss,c@((q*v**-.5)@q.T),occ,fun,orbital_tolerance=1e-11)
        values.append(trial.energy_hartree)
    numerical=(values[0]-2*values[1]+values[2])/step**2
    assert analytic==pytest.approx(numerical,abs=5e-7)


def test_anchor_changes_energy_and_apoc_metadata_but_preserves_derivatives():
    from matrix_gfn2.anchored_roks import FrozenROKSAnchor,AnchoredROKSResult
    ref=FrozenSpinReference(np.array([4]),np.arange(3),-.1*np.eye(3),'test',.9)
    spin=ShellSpinPolarization(np.eye(3),np.arange(3),ref.coupling_hartree)
    h=np.diag([-.8,-.4,-.2]);occ=np.array([[1,1,0],[1,0,1]])
    def callback(p):
        f,e,_=spin.evaluate(p)
        return h[None]+f,np.sum(p*h)+e
    r=run_roks_singlet(np.eye(3),np.eye(3),occ,callback)
    r=replace(r,spin_reference_fingerprint=ref.fingerprint,ground_energy_hartree=-2.,
        gradient_hartree_per_bohr=np.array([[.1,.2,.3]]),core_electrons=np.array([0]),
        reference_state=SimpleNamespace(atomic_numbers=np.array([4]),coordinates_bohr=np.zeros((1,3)),
                                       charge=0,model_id='toy'))
    anchor=FrozenROKSAnchor(ref,occ,np.zeros((1,3)),.3,.2,'frozen source test')
    result=AnchoredROKSResult(r,anchor)
    assert result.energy_hartree==pytest.approx(r.energy_hartree+.1)
    assert result.excitation_energy_hartree==pytest.approx(r.excitation_energy_hartree+.1)
    assert np.array_equal(result.gradient_hartree_per_bohr,r.gradient_hartree_per_bohr)
    state=result.to_apoc()
    assert state.energy_hartree==pytest.approx(result.energy_hartree)
    assert state.mixed.wavefunction.total_energy_hartree==pytest.approx(result.energy_hartree)
    assert anchor.fingerprint in state.mixed.wavefunction.model
    assert state.spin_squared==0
    with pytest.raises(ValueError,match='functional differ'):
        AnchoredROKSResult(replace(r,spin_reference_fingerprint=None),anchor)


def test_anchor_roundtrip_checks_every_parameter_and_reference_kind(tmp_path):
    from matrix_gfn2 import FrozenROKSAnchor,save_frozen_roks_anchor,load_frozen_roks_anchor
    ref=FrozenResponseReference(np.array([1,1]),np.arange(2),-.1*np.eye(2),'test',.9,
                               .03*np.eye(2),np.ones(2))
    a=FrozenROKSAnchor(ref,np.array([[0,1],[1,0]]),np.zeros((2,3)),.3,.2,'test',kind='relaxed_vertical')
    path=tmp_path/'anchor.npz';save_frozen_roks_anchor(path,a)
    b=load_frozen_roks_anchor(path)
    assert b.fingerprint==a.fingerprint and b.kind=='relaxed_vertical'
    assert replace(a,kind='frozen_orbitals').fingerprint!=a.fingerprint
    with np.load(path) as f:data={k:f[k] for k in f.files}
    data['coupling_hartree']=data['coupling_hartree']*.8
    np.savez_compressed(path,**data)
    with pytest.raises(ValueError,match='checksum'):load_frozen_roks_anchor(path)
