from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from matrix_gfn2.hybrid_spectrum import (match_orbital_spectrum,
    replace_tda_reference_gaps,compose_hybrid_excitation,correct_roks_excitation,
    OccupiedExtentScaling,with_occupied_extent_scaling)
from matrix_gfn2.roks import ROKSSingletResult


def mapping():
    perm=np.array([5,2,0,4,1,3]);ref=np.eye(6)[:,perm];ref[:,1]*=-1
    cross=np.eye(6)[1:5]
    return match_orbital_spectrum(np.eye(4),np.eye(4),[2,2,0,0],[-.5,-.3,.05,.2],
        ref,np.eye(6),np.array([2,2,2,0,0,0])[perm],
        np.array([-10.,-.6,-.4,.1,.3,1.])[perm],cross,reference_model='DFT/test')


def test_mapping_is_phase_permutation_invariant_and_separates_core_and_virtuals():
    m=mapping()
    assert np.allclose(m.reference_energies_hartree,[-.6,-.4,.1,.3])
    assert np.all(m.qualified)
    assert m.gap_correction(1,2).value_hartree==pytest.approx(.15)
    assert not m.reference_energies_hartree.flags.writeable


def test_ambiguous_mapping_and_invalid_cross_metric_cannot_silently_supply_a_gap():
    args=(np.eye(2),np.eye(2),[2,0],[-.5,.2],np.eye(4),np.eye(4),[2,2,0,0],[-1.,-.5,.1,.4])
    cross=np.array([[1,1,0,0],[0,0,1,1]])/np.sqrt(2)
    m=match_orbital_spectrum(*args,cross,reference_model='ambiguous')
    assert not np.any(m.qualified)
    with pytest.raises(ValueError,match='ambiguous'):m.gap_correction(0,1)
    with pytest.raises(ValueError,match='metric'):match_orbital_spectrum(*args,2*cross,reference_model='invalid')
    with pytest.raises(ValueError):m.gap_correction(-1,1)


def test_tda_replaces_configuration_diagonal_and_rediagonalizes_mixed_states():
    m=mapping();a=np.array([[.7,.12],[.12,.9]])
    result=replace_tda_reference_gaps(a,np.array([0,1]),np.array([2,2]),m)
    assert result[0,1]==a[0,1]
    assert np.allclose(np.diag(result-a),[.15,.15])
    # Different particle corrections change a mixing angle, not just root energies.
    result=replace_tda_reference_gaps(a,np.array([0,1]),np.array([2,3]),m)
    assert abs(np.linalg.eigh(result)[1][0,0])!=pytest.approx(abs(np.linalg.eigh(a)[1][0,0]))
    assert np.array_equal(a,np.array([[.7,.12],[.12,.9]]))


def test_energy_only_reference_cannot_inherit_source_gradient_or_hessian():
    jet=mapping().gap_correction(1,2)
    value=compose_hybrid_excitation(.8,jet,source_gradient=[.2],source_hessian=[[.4]])
    assert value.excitation_energy_hartree==pytest.approx(.95)
    assert value.gradient_hartree_per_bohr is None and value.hessian_hartree_per_bohr2 is None
    with pytest.raises(ValueError,match='gap-correction derivatives'):
        compose_hybrid_excitation(.8,jet,source_gradient=[.2],gradient=True)
    jet=mapping().gap_correction(1,2,gradient=[.3])
    with pytest.raises(ValueError,match='Hessian'):
        compose_hybrid_excitation(.8,jet,source_gradient=[.2],source_hessian=[[.4]],hessian=True)


def test_full_analytic_derivatives_match_independent_polynomial_differences():
    m=mapping();jet=m.gap_correction(1,2,gradient=[.3],hessian=[[.4]])
    r=compose_hybrid_excitation(.8,jet,source_gradient=[-.2],source_hessian=[[.6]],hessian=True)
    def energy(x):return .8-.2*x+.3*x*x+jet.value_hartree+.3*x+.2*x*x
    step=1e-4
    assert r.gradient_hartree_per_bohr[0]==pytest.approx((energy(step)-energy(-step))/(2*step),abs=1e-10)
    assert r.hessian_hartree_per_bohr2[0,0]==pytest.approx((energy(step)-2*energy(0)+energy(-step))/step**2,abs=1e-7)


def test_roks_uses_original_ground_spectrum_and_returns_gap_derivatives():
    m=mapping();state=SimpleNamespace(coefficients=np.eye(4),overlap=np.eye(4),
        alpha_occupations=np.array([1,1,0,0]),beta_occupations=np.array([1,1,0,0]),
        orbital_energies_hartree=np.array([-.5,-.3,.05,.2]))
    occ=np.array([[1,0,1,0],[1,1,0,0]])
    r=ROKSSingletResult(np.eye(4),occ,np.array([[1,1,1,0],[1,0,0,0]]),np.eye(4),
        np.zeros((2,4,4)),np.zeros((2,4,4)),-1.,0.,1,True,
        ground_energy_hartree=-2.,reference_state=state,gradient_hartree_per_bohr=np.array([[.1,.2,.3]]))
    corrected=correct_roks_excitation(r,m,1,2)
    assert corrected.excitation_energy_hartree==pytest.approx(1.15)
    jet=m.gap_correction(1,2,gradient=[.01,.02,.03],hessian=.1*np.eye(3))
    corrected=correct_roks_excitation(r,m,1,2,correction=jet,ground_gradient=[.01,.02,.03],
        excitation_hessian=.2*np.eye(3),hessian=True)
    assert np.allclose(corrected.gradient_hartree_per_bohr,[.1,.2,.3])
    assert np.allclose(corrected.hessian_hartree_per_bohr2,.3*np.eye(3))
    with pytest.raises(ValueError,match='different ground'):
        correct_roks_excitation(replace(r,reference_state=SimpleNamespace(**{**vars(state),'orbital_energies_hartree':np.array([-.6,-.3,.05,.2])})),m,1,2)


def test_global_reference_energy_offset_cancels_from_every_excitation_gap():
    m=match_orbital_spectrum(np.eye(4),np.eye(4),[2,2,0,0],[-.6,-.3,.1,.4],
        np.eye(4),np.eye(4),[2,2,0,0],np.array([-.6,-.3,.1,.4])+7.,np.eye(4),reference_model='shifted gauge')
    a=np.array([[.7,.12],[.12,.9]])
    corrected=replace_tda_reference_gaps(a,np.array([0,1]),np.array([2,3]),m)
    assert np.allclose(corrected,a,atol=1e-14,rtol=0)
    assert m.gap_correction(1,2).value_hartree==pytest.approx(0,abs=1e-14)


def test_one_reference_scale_is_shared_by_roks_gap_and_tda_diagonal():
    m=replace(mapping(),reference_energy_scale=.9)
    i=np.array([0,1]);a=np.array([2,3])
    se_gap=m.source_energies_hartree[a]-m.source_energies_hartree[i]
    ref_gap=m.reference_energies_hartree[a]-m.reference_energies_hartree[i]
    kernel=np.array([[.02,.12],[.12,-.03]])
    corrected=replace_tda_reference_gaps(np.diag(se_gap)+kernel,i,a,m)
    assert np.allclose(corrected,np.diag(.9*ref_gap)+kernel)
    for k in range(2):
        # A frozen gap plus arbitrary SE relaxation receives the same shift.
        result=compose_hybrid_excitation(se_gap[k]-.07,m.gap_correction(int(i[k]),int(a[k])))
        assert result.excitation_energy_hartree==pytest.approx(.9*ref_gap[k]-.07)
    shifted=replace(m,reference_energies_hartree=m.reference_energies_hartree+7.)
    assert np.allclose(shifted.configuration_corrections(i,a),m.configuration_corrections(i,a))
    for bad in (True,0.,-.5,np.inf,np.nan):
        with pytest.raises(ValueError,match='scale'):replace(m,reference_energy_scale=bad)


def test_scaled_surface_derivatives_include_scaled_reference_gap_response():
    m=replace(mapping(),reference_energy_scale=.9)
    # Independent SE excitation, SE gap and DFT gap Taylor coefficients.
    se=np.array([.8,-.2,.6]);sg=np.array([.35,.1,.2]);df=np.array([.5,.3,.4])
    correction=.9*df-sg
    jet=m.gap_correction(1,2,gradient=[correction[1]],hessian=[[correction[2]]])
    assert jet.value_hartree==pytest.approx(correction[0])
    result=compose_hybrid_excitation(se[0],jet,source_gradient=[se[1]],source_hessian=[[se[2]]],hessian=True)
    def energy(x):
        powers=np.array([1.,x,.5*x*x])
        return (se+.9*df-sg)@powers
    step=1e-4
    assert result.gradient_hartree_per_bohr[0]==pytest.approx((energy(step)-energy(-step))/(2*step),abs=1e-10)
    assert result.hessian_hartree_per_bohr2[0,0]==pytest.approx((energy(step)-2*energy(0)+energy(-step))/step**2,abs=1e-7)


def test_extent_scaling_defines_one_spectrum_for_all_transitions_and_both_methods():
    m=replace(mapping(),reference_energy_scale=.9)
    model=with_occupied_extent_scaling(m,[.7,1.,2.,3.],[1.,1.,2.,3.])
    expected=np.array([-.39,-.4,.1,.3])
    assert np.allclose(model.effective_reference_energies_hartree,expected)
    i=np.array([0,1,0,1]);a=np.array([2,2,3,3])
    source_gaps=m.source_energies_hartree[a]-m.source_energies_hartree[i]
    kernel=np.full((4,4),.02)
    result=replace_tda_reference_gaps(np.diag(source_gaps)+kernel,i,a,model)
    assert np.allclose(result,np.diag(.9*(expected[a]-expected[i]))+kernel)
    for k in range(len(i)):
        energy=compose_hybrid_excitation(source_gaps[k]-.07,model.gap_correction(int(i[k]),int(a[k])))
        assert energy.excitation_energy_hartree==pytest.approx(.9*(expected[a[k]]-expected[i[k]])-.07)
    assert np.array_equal(model.reference_energies_hartree,m.reference_energies_hartree)
    assert model.extent_scaling.weights[0]==pytest.approx(.7)
    assert not model.extent_scaling.source_radii_bohr.flags.writeable
    with pytest.raises(ValueError):with_occupied_extent_scaling(model,[1.]*4,[1.]*4)


def test_extent_spectrum_is_gauge_invariant_and_reduces_to_original_for_equal_extents():
    m=replace(mapping(),reference_energy_scale=.9)
    model=with_occupied_extent_scaling(m,[.7,1.,2.,3.],[1.,1.,2.,3.])
    shifted=replace(model,reference_energies_hartree=model.reference_energies_hartree+7.)
    assert np.allclose(shifted.effective_reference_energies_hartree,model.effective_reference_energies_hartree+7.)
    i=np.array([0,1]);a=np.array([2,3])
    assert np.allclose(shifted.configuration_corrections(i,a),model.configuration_corrections(i,a))
    unchanged=with_occupied_extent_scaling(m,[1.,2.,3.,4.],[1.,2.,3.,4.])
    assert np.allclose(unchanged.configuration_corrections(i,a),m.configuration_corrections(i,a))
    # No clipping kink when the source and reference radii cross equality.
    diffuse=with_occupied_extent_scaling(m,[2.]*4,[1.]*4)
    expected=m.reference_energies_hartree.copy()
    expected[:2]=.1+2*(expected[:2]-.1)
    assert np.allclose(diffuse.effective_reference_energies_hartree,expected)


def test_extent_descriptor_rejects_invalid_radii_and_unqualified_anchor():
    for source,reference in [([0.,1.],[1.,1.]),([1.,np.nan],[1.,1.]),([1.],[1.,2.]),([[1.]],[[1.]])]:
        with pytest.raises(ValueError):OccupiedExtentScaling(source,reference)
    m=mapping()
    with pytest.raises(ValueError):with_occupied_extent_scaling(m,[1.]*3,[1.]*3)
    overlap=m.overlaps.copy();overlap[2]=.3
    with pytest.raises(ValueError,match='ambiguous'):
        with_occupied_extent_scaling(replace(m,overlaps=overlap),[1.]*4,[1.]*4)
    degenerate=m.reference_energies_hartree.copy();degenerate[3]=degenerate[2]
    with pytest.raises(ValueError,match='isolated'):
        with_occupied_extent_scaling(replace(m,reference_energies_hartree=degenerate),[1.]*4,[1.]*4)


def test_fractional_extent_power_is_explicit_shared_and_never_fitted_by_the_api():
    m=replace(mapping(),reference_energy_scale=.8706205766726199)
    extent=with_occupied_extent_scaling(m,[.7,1.,2.,3.],[1.,1.,2.,3.],power=1.5)
    assert extent.extent_scaling.weights[0]==pytest.approx(.7**1.5)
    # Transitions share one modified occupied level and retain virtual spacing.
    c0=extent.gap_correction(0,2).value_hartree
    c1=extent.gap_correction(0,3).value_hartree
    expected=m.reference_energy_scale*(m.reference_energies_hartree[3]-m.reference_energies_hartree[2])-(m.source_energies_hartree[3]-m.source_energies_hartree[2])
    assert c1-c0==pytest.approx(expected)
    matrix=replace_tda_reference_gaps(np.eye(2),np.array([0,0]),np.array([2,3]),extent)
    assert np.allclose(np.diag(matrix),[1+c0,1+c1])
    for bad in [True,0.,-1.,np.inf,np.nan]:
        with pytest.raises(ValueError,match='power'):with_occupied_extent_scaling(m,[1.]*4,[1.]*4,power=bad)
