"""Remote molecular regression for the SCC-independent absolute functional."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import importlib.util,socket
import numpy as np
import pytest
from scipy.linalg import expm

@pytest.fixture
def molecular():
    assert socket.getfqdn()=='oracle.sns.it','electronic qualification runs only on Oracle'
    spec=importlib.util.spec_from_file_location('direct_probe',Path(__file__).with_name('probe.py'))
    p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
    data,retained=p.retained('nitroxyl')
    calc=p.GFN2Calculator(data['atoms'],temperature_kelvin=0.)
    return p,data,retained,calc


def test_exact_absolute_functional_and_hessian_without_any_ground_scc(molecular,tmp_path):
    from matrix_link.qmqm_component import CappedQMComponent
    p,data,old,calc=molecular
    def forbidden(*a,**kw):raise AssertionError('ground SCC invoked in direct ROKS path')
    calc.evaluate=forbidden
    backend=replace(old.backend,native_roks_composite_reference_policy='geometry_only',native_roks_composite_component_tracking='active_pair',native_roks_composite_following='local_csf_adaptive_index')
    comp=CappedQMComponent(backend,SimpleNamespace(_calculator=calc),tmp_path)
    comp.reference=old.reference;comp.anchor=old.anchor
    retained=old.reference[0];xyz=retained.components[0].reference_state.coordinates_bohr/p.ANGSTROM_TO_BOHR
    point=comp.evaluate(xyz,compute_hessian=True)
    assert point.energy_hartree==pytest.approx(retained.energy_hartree,abs=5e-12)
    assert np.max(abs(point.gradient_hartree_per_bohr-retained.gradient_hartree_per_bohr.ravel()))<1e-10
    assert point.scf_state[0].excitation_energy_hartree is None
    assert point.execution['stationary_reference']['ground_scc_evaluations']==0
    direction=-retained.gradient_hartree_per_bohr;direction/=np.linalg.norm(direction)
    for step in [2e-4,1e-4]:
        minus=comp.evaluate(xyz-step/p.ANGSTROM_TO_BOHR*direction)
        plus=comp.evaluate(xyz+step/p.ANGSTROM_TO_BOHR*direction)
        fd=(plus.gradient_hartree_per_bohr-minus.gradient_hartree_per_bohr)/(2*step)
        assert np.max(abs(fd-point.hessian_hartree_per_bohr2@direction.ravel()))<1e-7


def test_component_active_pair_guard_excludes_closed_spectator_rotations(molecular):
    from matrix_gfn2 import combine_roks_results
    from matrix_gfn2.composite_roks import CompositeROKSStateMismatch
    p,data,retained,calc=molecular
    original=retained.reference[0].components[0];n=len(original.overlap)
    closed=np.flatnonzero(original.occupations.sum(axis=0)==2)
    virtual=np.flatnonzero(original.occupations.sum(axis=0)==0)
    generator=np.zeros((n,n))
    for i,j in zip(closed,virtual):generator[i,j]=1.;generator[j,i]=-1.
    # Exactly stationary toy constant Hamiltonians isolate the overlap algebra.
    common=dict(fock_mixed=np.zeros_like(original.fock_mixed),fock_triplet=np.zeros_like(original.fock_triplet),orbital_residual=0.,orbital_saddle_index=0,target_orbital_saddle_index=0)
    a=replace(original,spin_reference_fingerprint='toy.left',**common)
    b=replace(original,common_coefficients=original.common_coefficients@expm(generator),spin_reference_fingerprint='toy.right',**common)
    with pytest.raises(CompositeROKSStateMismatch):combine_roks_results(a,b)
    result=combine_roks_results(a,b,component_tracking='active_pair')
    assert result.component_active_pair_overlap==pytest.approx(1.)
    assert result.component_csf_overlap<.7


def test_geometry_reference_rejects_another_hamiltonian(molecular):
    from matrix_gfn2.orbital_geometry_reference import geometry_orbital_reference
    p,data,retained,calc=molecular
    state=retained.reference[0].components[0].reference_state
    with pytest.raises(ValueError,match='Hamiltonian'):
        geometry_orbital_reference(calc,state.coordinates_bohr/p.ANGSTROM_TO_BOHR,replace(state,model_id='another'))


def test_restricted_ground_derivatives_without_scc(molecular):
    from matrix_gfn2.restricted_ground import evaluate_restricted_ground, restricted_ground_analytic_hessian
    p, data, retained, calc = molecular
    calc.evaluate = lambda *a, **kw: (_ for _ in ()).throw(AssertionError('SCC invoked'))
    template = retained.reference[0].components[0].reference_state
    xyz = template.coordinates_bohr/p.ANGSTROM_TO_BOHR
    for ref in [None, p.load_response_reference(data['response_paths'][0])]:
        result = evaluate_restricted_ground(calc, xyz, template=template, spin_reference=ref, gradient=True)
        assert result.orbital_residual < 1e-10
        assert result.minimum_orbital_curvature_hartree > 1e-7
        assert abs(result.spin_squared) < 1e-10
        h = restricted_ground_analytic_hessian(calc, xyz, reference_result=result, spin_reference=ref)
        direction = -result.gradient_hartree_per_bohr
        direction /= np.linalg.norm(direction)
        errors = []
        for step in [2e-4, 1e-4]:
            kwargs = dict(template=template, spin_reference=ref, reference_result=result, gradient=True)
            a = evaluate_restricted_ground(calc, xyz-step/p.ANGSTROM_TO_BOHR*direction, **kwargs)
            b = evaluate_restricted_ground(calc, xyz+step/p.ANGSTROM_TO_BOHR*direction, **kwargs)
            assert abs((b.energy_hartree-a.energy_hartree)/(2*step)-np.sum(result.gradient_hartree_per_bohr*direction)) < 1e-7
            errors.append(np.max(abs((b.gradient_hartree_per_bohr-a.gradient_hartree_per_bohr).ravel()/(2*step)-h.hessian_hartree_per_bohr2@direction.ravel())))
        assert errors[1] < 1e-7
        assert errors[1] < .35*errors[0]


@pytest.mark.parametrize('following',['local_csf_adaptive_index','character_newton_index'])
def test_ground_referenced_link_derivatives_and_restart(molecular,tmp_path,following):
    from matrix_link.qmqm_component import CappedQMComponent
    from matrix_link.composite_roks_component import persist_accepted
    from matrix_gfn2.ground_referenced_roks import GroundReferencedROKSResult
    p,data,old,calc=molecular
    calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('SCC invoked'))
    backend=replace(old.backend,native_roks_composite_reference_policy='geometry_only',
        native_roks_composite_component_tracking='active_pair',native_roks_composite_following=following,
        native_roks_composite_ground_surface='native_plus_gaps')
    comp=CappedQMComponent(backend,SimpleNamespace(_calculator=calc),tmp_path)
    comp.reference=old.reference;comp.anchor=old.anchor
    xyz=old.reference[0].components[0].reference_state.coordinates_bohr/p.ANGSTROM_TO_BOHR
    point=comp.evaluate(xyz,compute_hessian=True)
    assert isinstance(point.scf_state[0],GroundReferencedROKSResult)
    assert point.scf_state[0].spectrum_qualified
    direction=np.cos(np.arange(xyz.size)+.4).reshape(xyz.shape)
    direction-=direction.mean(axis=0);direction/=np.linalg.norm(direction)
    step=1e-4
    a=comp.evaluate(xyz-step/p.ANGSTROM_TO_BOHR*direction)
    b=comp.evaluate(xyz+step/p.ANGSTROM_TO_BOHR*direction)
    assert abs((b.energy_hartree-a.energy_hartree)/(2*step)-point.gradient_hartree_per_bohr@direction.ravel())<1e-7
    assert np.max(abs((b.gradient_hartree_per_bohr-a.gradient_hartree_per_bohr)/(2*step)-point.hessian_hartree_per_bohr2@direction.ravel()))<1e-6
    persist_accepted(comp,point.scf_state)
    restored=CappedQMComponent(backend,SimpleNamespace(_calculator=calc),tmp_path)
    assert len(restored.reference[0].grounds)==3
    assert restored.reference[0].energy_hartree==pytest.approx(point.energy_hartree,abs=1e-12)
    repeated=restored.evaluate(xyz)
    assert repeated.energy_hartree==pytest.approx(point.energy_hartree,abs=1e-12)
    assert repeated.execution['stationary_reference']['ground_scc_evaluations']==0
    assert repeated.execution['ground_surface']=='native_plus_gaps'
    with pytest.raises(ValueError,match='contract'):
        CappedQMComponent(replace(backend,native_roks_composite_ground_surface=None),SimpleNamespace(_calculator=calc),tmp_path)


def test_ground_referenced_link_rejects_real_inverted_spectrum(molecular,tmp_path):
    from matrix_link.qmqm_component import CappedQMComponent
    from matrix_apoc import StateContinuityError
    from matrix_gfn2 import combine_roks_results
    from matrix_gfn2.deltascf_derivatives import load_derivative_point
    p,_,_,_=molecular
    root=p.BASE/'results/acetaldehyde'
    endpoint=root/'ground_replacement_minimum'
    data=__import__('json').loads((root/'prepared.json').read_text())
    with np.load(endpoint/'accepted_geometry.npz') as d:xyz=d['xyz'].copy()
    states=[load_derivative_point(endpoint/f'excited_{i}.npz',request_hash='ground_replacement_v1',coordinates_angstrom=xyz) for i in range(2)]
    retained=combine_roks_results(*states,component_tracking='active_pair')
    backend=p.QMScanBackend(name='matrix_gfn2',electronic_state=1,native_excited_method='roks_50_50',
        deltascf_promotion=tuple(data['promotion']),native_roks_composite_response_paths=tuple(data['response_paths']),
        native_roks_composite_saddle_indices=tuple(data['saddle_indices']),native_roks_composite_reference_policy='geometry_only',
        native_roks_composite_component_tracking='active_pair',native_roks_composite_following='local_csf',
        native_roks_composite_ground_surface='native_plus_gaps')
    calc=p.GFN2Calculator(data['atoms'],temperature_kelvin=0.)
    calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('SCC invoked'))
    component=CappedQMComponent(backend,SimpleNamespace(_calculator=calc),tmp_path)
    component.reference=(retained,tuple(r.to_apoc(1) for r in states));component.anchor=component.reference
    with pytest.raises(StateContinuityError,match='nonpositive component excitation gap'):
        component.evaluate(xyz)
    assert not (tmp_path/'roks_50_50_accepted/accepted.json').exists()


def test_positive_density_transport_has_analytic_second_metric_and_frame_jets(molecular):
    from matrix_gfn2.lowdin_operator_jet import LowdinOperatorJet
    p,_,retained,_=molecular
    state=retained.reference[0].components[0].reference_state
    rng=np.random.default_rng(748);n=len(state.overlap);nr=state.coordinates_bohr.size
    a=rng.normal(size=(3,n,n));a=.5*(a+a.transpose(0,2,1))
    linear=rng.normal(size=(n,n));linear=.015*(linear+linear.T)
    quadratic=rng.normal(size=(n,n));quadratic=.005*(quadratic+quadratic.T)
    weights=np.cos(np.arange(nr)+.8);weights/=np.linalg.norm(weights)
    direction=np.sin(np.arange(nr)+.4);direction/=np.linalg.norm(direction)
    def jet(displacement):
        z=weights@displacement
        s=state.overlap+z*linear+z*z*quadratic
        ds=weights[:,None,None]*(linear+2*z*quadratic)
        d2s=2*weights[:,None,None,None]*weights[None,:,None,None]*quadratic
        moved=SimpleNamespace(**vars(state))
        moved.coordinates_bohr=state.coordinates_bohr+displacement.reshape(-1,3);moved.overlap=s
        return LowdinOperatorJet(moved,state.coordinates_bohr,a,overlap_first=ds,overlap_second=d2s)
    center=jet(np.zeros(nr));first=center.first()
    exact_first=np.einsum('r,raij->aij',direction,first)
    exact_second=np.array([sum(direction[j]*center.second(i,j) for j in range(nr)) for i in range(nr)])
    errors=[]
    for step in [1e-4,5e-5]:
        minus,plus=jet(-step*direction),jet(step*direction)
        assert np.max(abs((plus.operators-minus.operators)/(2*step)-exact_first))<1e-7
        errors.append(np.max(abs((plus.first()-minus.first())/(2*step)-exact_second)))
    assert errors[1]<1e-7 and errors[1]<.35*errors[0]
    # The square-root congruence is positive even away from the reference.
    moved=jet(.1*direction);u=moved.transport.u;l=moved.root
    q=rng.normal(size=(n,3));density=q@q.T
    transformed=u.T@l@density@l@u
    assert np.linalg.eigvalsh(transformed).min()>-1e-12
    # Check the adjoint metric force against the full directional operator jet
    # after subtracting its explicit frame rotation part.
    fields=rng.normal(size=(2,3));densities=np.array([density,.3*density])
    metric=center.metric_potential(densities,fields)
    frame=center.root@np.einsum('r,raij->aij',direction,center.transport.first())@center.root
    directional=exact_first-frame
    assert np.einsum('ij,ij->',metric,(weights@direction)*linear)==pytest.approx(
        np.einsum('sa,aij,sij->',fields,directional,densities),abs=1e-10)


def test_newton_follows_and_certifies_a_real_electronic_saddle(molecular):
    from matrix_gfn2.deltascf_derivatives import load_derivative_point
    from matrix_gfn2.deltascf_analytic_hessian import evaluate_gfn2_deltascf_analytic_hessian
    p,data,_,calc=molecular
    cp=p.BASE/'results/nitroxyl/fr_index1.npz'
    with np.load(cp) as d:xyz=d['coordinates_angstrom'].copy()
    old=load_derivative_point(cp,request_hash='fr_index1',coordinates_angstrom=xyz)
    calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('SCC invoked'))
    ref=p.load_response_reference(data['response_paths'][0])
    result=p.evaluate_gfn2_roks_singlet(calc,xyz,*old.occupations,spin_reference=ref,
        reference_result=old,reference_policy='geometry_only',orbital_solver='newton',
        certify_index=True,orbital_tolerance=1e-10,gradient=True)
    assert result.orbital_saddle_index==result.target_orbital_saddle_index==1
    h=evaluate_gfn2_deltascf_analytic_hessian(calc,xyz,*result.occupations,
        mode='roks',reference_result=result,spin_reference=ref)
    assert np.count_nonzero(np.linalg.eigvalsh(h.orbital_hessian_hartree)<-1e-7)==1
    assert abs(result.energy_hartree-old.energy_hartree)<1e-11
