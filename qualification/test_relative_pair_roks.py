"""Fixed relative-state manifold: exact vertical preservation and nuclear KKT."""
from pathlib import Path
import json
import numpy as np
import pytest
from test_shared_pair_roks import problem


def test_relative_manifold_preserves_assigned_vertical_energy(problem):
    from matrix_gfn2 import evaluate_relative_pair_roks,relative_active_targets
    calc,xyz,refs,parents=problem
    result=evaluate_relative_pair_roks(calc,xyz,refs,initial_components=parents)
    assert abs(result.energy_hartree-.5*sum(p.energy_hartree for p in parents))<1e-10
    assert np.max(abs(result.multipliers))<1e-7
    assert np.max(abs(result.target_squared_overlaps-relative_active_targets(parents)))<1e-12
    assert result.orbital_residual<1e-9 and result.spin_squared==0.
    with pytest.raises(ValueError,match='cannot change'):
        evaluate_relative_pair_roks(calc,xyz,refs,reference_result=result,
            target_squared_overlaps=result.target_squared_overlaps*.99)


def test_relative_manifold_coupled_nuclear_hessian_and_restart(problem,tmp_path):
    from matrix_gfn2 import (evaluate_relative_pair_roks,relative_active_targets,
        reference_roks_to_native_ground,ground_referenced_roks_analytic_hessian,
        save_relative_pair_roks,load_relative_pair_roks)
    from matrix_gfn2.deltascf_derivatives import load_derivative_point
    from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
    calc,xyz,refs,parents=problem
    folder=Path('/home/enzo/MOUSE-roks-local-20261030/results/formamide')
    anchors=[]
    for i in range(2):
        with np.load(folder/f'anchor_{i}.npz') as d:geom=d['coordinates_angstrom'].copy();request=str(d['request_hash'])
        anchors.append(load_derivative_point(folder/f'anchor_{i}.npz',request_hash=request,coordinates_angstrom=geom,require_gradient=False))
    result=evaluate_relative_pair_roks(calc,xyz,refs,initial_components=parents,anchor_components=anchors,gradient=True)
    assert result.orbital_residual<1e-9 and np.max(abs(result.multipliers))>.1
    assert all(not p.converged and p.orbital_saddle_index is None for p in result.components)
    assert max(result.component_unconstrained_residuals)>1e-3
    assert np.max(abs(relative_active_targets(result.components)-relative_active_targets(anchors)))<1e-9
    save_relative_pair_roks(tmp_path/'relative.npz',result,request_hash='relative_nuclear_test')
    restored=load_relative_pair_roks(tmp_path/'relative.npz',request_hash='relative_nuclear_test')
    assert restored.energy_hartree==result.energy_hartree
    assert np.array_equal(restored.target_squared_overlaps,result.target_squared_overlaps)
    with pytest.raises(ValueError,match='another request'):
        load_relative_pair_roks(tmp_path/'relative.npz',request_hash='independent_orbital_surface')
    total=reference_roks_to_native_ground(calc,xyz,refs,excited=restored)
    h=ground_referenced_roks_analytic_hessian(calc,xyz,refs,reference_result=total)
    assert h.excited.maximum_response_residual<1e-10
    direction=np.sin(np.arange(xyz.size)+.3).reshape(xyz.shape);direction-=direction.mean(axis=0);direction/=np.linalg.norm(direction)
    errors=[]
    for step in [2e-4,1e-4]:
        points=[]
        for sign in [-1,1]:
            geom=xyz+sign*step/ANGSTROM_TO_BOHR*direction
            excited=evaluate_relative_pair_roks(calc,geom,refs,reference_result=restored,gradient=True)
            points.append(reference_roks_to_native_ground(calc,geom,refs,excited=excited,reference_result=total))
        a,b=points
        assert abs((b.energy_hartree-a.energy_hartree)/(2*step)-np.sum(total.gradient_hartree_per_bohr*direction))<1e-7
        errors.append(np.max(abs((b.gradient_hartree_per_bohr-a.gradient_hartree_per_bohr).ravel()/(2*step)-h.hessian_hartree_per_bohr2@direction.ravel())))
    assert errors[1]<1e-7 and errors[1]<.4*errors[0]
