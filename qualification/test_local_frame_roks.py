"""Remote molecular qualification of static local response-frame transport."""
import importlib.util,socket,json
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

@pytest.fixture
def setup():
    assert socket.getfqdn()=='oracle.sns.it'
    base=Path('/home/enzo/MOUSE-roks-direct-20261030')
    spec=importlib.util.spec_from_file_location('p',base/'source/qualification/probe.py')
    p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
    return p,base


def local_points(p,base,tmp_path,name='acetaldehyde',heavy_first=False):
    from matrix_gfn2 import FrozenLocalFrameResponseReference,save_response_reference
    from matrix_gfn2.deltascf_derivatives import load_derivative_point
    data=json.loads((base/'results'/name/'prepared.json').read_text());refs=[];results=[];paths=[]
    calc=p.GFN2Calculator(data['atoms'],temperature_kelvin=0.)
    calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('SCC invoked'))
    for i,path in enumerate(data['response_paths']):
        ref=FrozenLocalFrameResponseReference.from_reference(p.load_response_reference(path),heavy_first=heavy_first);dest=tmp_path/f'response_{i}.npz'
        save_response_reference(dest,ref);restored=p.load_response_reference(dest)
        assert isinstance(restored,FrozenLocalFrameResponseReference) and restored.fingerprint==ref.fingerprint
        assert np.array_equal(restored.frame_clusters,ref.frame_clusters)
        refs.append(restored);paths.append(dest)
        cp=base/'results'/name/'local_frames'/f'relaxed_{i}.npz'
        with np.load(cp) as d:xyz=d['coordinates_angstrom'].copy()
        old=load_derivative_point(cp,request_hash='local_frame_probe_v1',coordinates_angstrom=xyz,require_gradient=False)
        seed=replace(old,spin_reference_fingerprint=ref.fingerprint,spin_frame_clusters=ref.frame_clusters)
        ctx=p.prepare_gfn2_deltascf(calc,xyz,orbital_reference=seed.reference_state,spin_reference=ref)
        fm,em=ctx.fock_and_energy(seed.densities());ft,et=ctx.fock_and_energy(seed.densities(triplet=True))
        if not heavy_first:
            assert max(np.max(abs(fm-seed.fock_mixed)),np.max(abs(ft-seed.fock_triplet)))<1e-11
            assert abs(2*em-et-seed.energy_hartree)<1e-11
        else:
            from matrix_gfn2.roks_orbital_response import run_roks_orbital_newton
            stationary=run_roks_orbital_newton(ctx,seed.common_coefficients,seed.occupations,
                orbital_tolerance=1e-10,maximum_evaluations=150,maximum_step=.2,certify_index=True)
            seed=replace(stationary,reference_state=ctx.state,core_electrons=seed.core_electrons,
                spin_reference_fingerprint=ref.fingerprint,spin_frame_clusters=ref.frame_clusters)
        result=p.evaluate_gfn2_roks_singlet(calc,xyz,*seed.occupations,spin_reference=ref,
            reference_result=seed,reference_policy='geometry_only',orbital_solver='gmf',
            saddle_index=seed.orbital_saddle_index,orbital_tolerance=1e-10,gradient=True)
        results.append(result)
    return data,calc,refs,results,paths,xyz


@pytest.mark.parametrize('heavy_first',[False,True])
def test_local_response_nuclear_gradient_and_hessian_at_nonplanar_geometry(setup,tmp_path,heavy_first):
    from matrix_gfn2.deltascf_analytic_hessian import evaluate_gfn2_deltascf_analytic_hessian
    p,base=setup;_,calc,refs,points,_,xyz=local_points(p,base,tmp_path,heavy_first=heavy_first)
    point=points[0];ref=refs[0]
    h=evaluate_gfn2_deltascf_analytic_hessian(calc,xyz,*point.occupations,
        reference_result=point,spin_reference=ref,mode='roks')
    direction=np.sin(np.arange(xyz.size)+.2).reshape(xyz.shape);direction-=direction.mean(axis=0);direction/=np.linalg.norm(direction)
    errors=[]
    for step in [2e-4,1e-4]:
        values=[]
        for sign in [-1,1]:
            values.append(p.evaluate_gfn2_roks_singlet(calc,xyz+sign*step/p.ANGSTROM_TO_BOHR*direction,
                *point.occupations,spin_reference=ref,reference_result=point,reference_policy='geometry_only',
                orbital_solver='gmf',saddle_index=point.orbital_saddle_index,orbital_tolerance=1e-10,gradient=True))
        a,b=values
        assert abs((b.energy_hartree-a.energy_hartree)/(2*step)-np.sum(point.gradient_hartree_per_bohr*direction))<1e-7
        errors.append(np.max(abs((b.gradient_hartree_per_bohr-a.gradient_hartree_per_bohr).ravel()/(2*step)-h.hessian_hartree_per_bohr2@direction.ravel())))
    assert errors[1]<1e-7 and errors[1]<.4*errors[0]


def test_hno_local_transport_is_identical_to_its_three_atom_global_frame(setup):
    from matrix_gfn2 import FrozenLocalFrameResponseReference
    from matrix_gfn2.deltascf_analytic_hessian import evaluate_gfn2_deltascf_analytic_hessian
    p,_=setup;data,retained=p.retained('nitroxyl');old=retained.reference[0].components[0]
    calc=p.GFN2Calculator(data['atoms'],temperature_kelvin=0.)
    calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('SCC invoked'))
    ref=p.load_response_reference(data['response_paths'][0]);local=FrozenLocalFrameResponseReference.from_reference(ref)
    assert np.all(local.frame_clusters==np.arange(3)[None,:])
    seed=replace(old,spin_reference_fingerprint=local.fingerprint,spin_frame_clusters=local.frame_clusters)
    xyz=old.reference_state.coordinates_bohr/p.ANGSTROM_TO_BOHR
    result=p.evaluate_gfn2_roks_singlet(calc,xyz,*old.occupations,spin_reference=local,
        reference_result=seed,reference_policy='geometry_only',orbital_solver='gmf',saddle_index=old.orbital_saddle_index,
        orbital_tolerance=1e-10,gradient=True)
    assert abs(result.energy_hartree-old.energy_hartree)<1e-11
    assert np.max(abs(result.gradient_hartree_per_bohr-old.gradient_hartree_per_bohr))<1e-11
    hg=evaluate_gfn2_deltascf_analytic_hessian(calc,xyz,*old.occupations,reference_result=old,spin_reference=ref,mode='roks')
    hl=evaluate_gfn2_deltascf_analytic_hessian(calc,xyz,*result.occupations,reference_result=result,spin_reference=local,mode='roks')
    assert np.max(abs(hg.hessian_hartree_per_bohr2-hl.hessian_hartree_per_bohr2))<1e-11


def test_local_frame_link_restart_preserves_model_and_state_frames(setup,tmp_path):
    from matrix_gfn2 import combine_roks_results
    from matrix_link.qmqm_component import CappedQMComponent
    from matrix_link.scan import QMScanBackend
    p,base=setup;data,calc,refs,points,paths,xyz=local_points(p,base,tmp_path)
    backend=QMScanBackend(name='matrix_gfn2',electronic_state=1,native_excited_method='roks_50_50',
        deltascf_promotion=tuple(data['promotion']),native_roks_composite_response_paths=tuple(paths),
        native_roks_composite_saddle_indices=tuple(data['saddle_indices']),native_roks_composite_reference_policy='geometry_only',
        native_roks_composite_component_tracking='active_pair',native_roks_composite_following='character_adaptive_index',
        native_roks_composite_ground_surface='native_plus_gaps')
    run=tmp_path/'link';component=CappedQMComponent(backend,SimpleNamespace(_calculator=calc),run)
    composite=combine_roks_results(*points,component_tracking='active_pair')
    component.reference=(composite,tuple(r.to_apoc(1) for r in points));component.anchor=component.reference
    point=component.evaluate(xyz);component.accept(point)
    assert all(g>0 for g in point.scf_state[0].component_gaps_hartree)
    restored=CappedQMComponent(backend,SimpleNamespace(_calculator=calc),run)
    for r,ref in zip(restored.reference[0].components,refs):
        assert np.array_equal(r.spin_frame_clusters,ref.frame_clusters)
    repeated=restored.evaluate(xyz)
    assert abs(repeated.energy_hartree-point.energy_hartree)<1e-11
    assert np.max(abs(repeated.gradient_hartree_per_bohr-point.gradient_hartree_per_bohr))<1e-10


def test_heavy_frame_does_not_follow_hydrogen_internal_rotation(setup):
    from matrix_gfn2 import FrozenLocalFrameResponseReference
    from matrix_gfn2.local_frame_response import LocalTransportedOperators
    from matrix_gfn2.deltascf_derivatives import load_derivative_point
    p,base=setup
    data=json.loads((base/'results/acetaldehyde/prepared.json').read_text())
    global_ref=p.load_response_reference(data['response_paths'][0])
    ref=FrozenLocalFrameResponseReference.from_reference(global_ref,heavy_first=True)
    cp=base/'results/acetaldehyde/local_frames/relaxed_0.npz'
    with np.load(cp) as d:xyz=d['coordinates_angstrom'].copy()
    state=load_derivative_point(cp,request_hash='local_frame_probe_v1',coordinates_angstrom=xyz,require_gradient=False).reference_state
    moved=SimpleNamespace(**vars(state));moved.coordinates_bohr=state.coordinates_bohr.copy()
    hydrogen=np.flatnonzero(state.atomic_numbers==1)
    moved.coordinates_bohr[hydrogen]+=np.sin(np.arange(3*len(hydrogen))).reshape(-1,3)*.1
    operators=np.eye(len(state.overlap))[None]
    a=LocalTransportedOperators(state,ref.anchor_coordinates_bohr,operators,ref.frame_clusters)
    b=LocalTransportedOperators(moved,ref.anchor_coordinates_bohr,operators,ref.frame_clusters)
    assert np.max(abs(a.u-b.u))<1e-14
    assert np.max(abs(b.du[np.concatenate([3*hydrogen+i for i in range(3)])]))==0
