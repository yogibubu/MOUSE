"""Checkpoint boundary: energy jobs can resume without weakening Hessian guards."""
from dataclasses import replace
import numpy as np
import pytest
from matrix_gfn2 import GFN2Calculator,evaluate_gfn2_deltascf
from matrix_gfn2.deltascf_derivatives import save_derivative_point,load_derivative_point


def test_energy_only_checkpoint_resumes_explicitly_but_default_still_requires_gradient(tmp_path):
    xyz=np.array([[0.,0.,0.],[0.,0.,.74]])
    calc=GFN2Calculator(['H','H'],temperature_kelvin=0.)
    ground=calc.evaluate(xyz,retain_orbitals=True,compute_cm5=False,compute_bond_orders=False)
    s=ground.orbital_state
    result=evaluate_gfn2_deltascf(calc,xyz,s.alpha_occupations,s.beta_occupations,
        ground_state=ground,gradient=False,selection='aufbau')
    assert result.converged and result.gradient_hartree_per_bohr is None
    path=tmp_path/'energy.npz'
    save_derivative_point(path,result,request_hash='energy-only',coordinates_angstrom=xyz)
    with pytest.raises(ValueError,match='gradient point'):
        load_derivative_point(path,request_hash='energy-only',coordinates_angstrom=xyz)
    resumed=load_derivative_point(path,request_hash='energy-only',coordinates_angstrom=xyz,require_gradient=False)
    assert resumed.energy_hartree==result.energy_hartree
    assert resumed.gradient_hartree_per_bohr is None
    assert np.array_equal(resumed.coefficients,result.coefficients)
    assert np.array_equal(resumed.reference_state.coefficients,s.coefficients)
    assert np.array_equal(resumed.reference_state.basis.primitive_exponents,s.basis.primitive_exponents)
    with pytest.raises(ValueError,match='does not match'):
        load_derivative_point(path,request_hash='different',coordinates_angstrom=xyz,require_gradient=False)
    with pytest.raises(ValueError,match='boolean'):
        load_derivative_point(path,request_hash='energy-only',coordinates_angstrom=xyz,require_gradient='false')
    for invalid in [replace(result,converged=False),replace(result,gradient_hartree_per_bohr=np.full((2,3),np.nan))]:
        save_derivative_point(path,invalid,request_hash='energy-only',coordinates_angstrom=xyz)
        with pytest.raises(ValueError,match='gradient point'):
            load_derivative_point(path,request_hash='energy-only',coordinates_angstrom=xyz,require_gradient=False)
