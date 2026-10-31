"""Explicit experimental orbital-optimized components for capped QM/QM'.

Only accepted optimizer points advance the reference in CappedQMComponent.
Promotions seed a state once; subsequent trials use projected accepted orbitals
and APOC determinant/CSF overlaps. No TDA root number is used as an orbital label.
"""
import numpy as np


def evaluate_component(component, coordinates, compute_hessian, ground, diagnostics):
    from matrix_gfn2 import (evaluate_gfn2_deltascf, evaluate_gfn2_roks_singlet,
                             evaluate_gfn2_deltascf_analytic_hessian)
    from matrix_apoc import determinant_overlap, singlet_csf_overlap
    from .scan import PointEvaluationResult

    backend = component.backend
    if backend.native_excited_method == 'roks_50_50':
        from .composite_roks_component import evaluate_composite_component
        return evaluate_composite_component(component,coordinates,compute_hessian,ground,diagnostics)
    calc = component.calculator._calculator
    mode = backend.native_excited_method
    frozen_anchor=None
    if backend.native_roks_anchor_path is not None:
        import hashlib
        from matrix_gfn2 import load_frozen_roks_anchor
        if hashlib.sha256(backend.native_roks_anchor_path.read_bytes()).hexdigest()!=backend.native_roks_anchor_sha256:
            raise ValueError('the frozen ROKS anchor changed during the calculation')
        frozen_anchor=load_frozen_roks_anchor(backend.native_roks_anchor_path)
    response_reference=None
    if backend.native_roks_response_path is not None:
        import hashlib
        from matrix_gfn2 import load_response_reference
        if hashlib.sha256(backend.native_roks_response_path.read_bytes()).hexdigest()!=backend.native_roks_response_sha256:
            raise ValueError('the no-offset ROKS response changed during the calculation')
        response_reference=load_response_reference(backend.native_roks_response_path)
    state = ground.orbital_state
    reference = None if component.reference is None else component.reference[0]
    if reference is None:
        alpha, beta = state.alpha_occupations.copy(), state.beta_occupations.copy()
        hole, particle = backend.deltascf_promotion
        if (max(hole, particle) >= len(alpha) or alpha[hole] != 1 or alpha[particle] != 0):
            raise ValueError('requested orbital promotion is not occupied -> virtual')
        alpha[hole], alpha[particle] = 0., 1.
    else:
        alpha, beta = reference.occupations
    options = dict(ground_state=ground, reference_result=reference, gradient=True,
                   minimum_state_overlap=backend.state_tracking_minimum_overlap,
                   orbital_tolerance=1e-10)
    if response_reference is not None:options['spin_reference']=response_reference
    if mode == 'roks':
        if backend.orbital_saddle_index is None:
            options.update(rechart_interval=8, maximum_evaluations=400)
        else:
            options.update(orbital_solver='gmf',saddle_index=backend.orbital_saddle_index,
                           maximum_iterations=100)
        if frozen_anchor is None:
            result = evaluate_gfn2_roks_singlet(calc, coordinates, alpha, beta, **options)
        else:
            from matrix_gfn2 import evaluate_anchored_roks
            if not np.array_equal(frozen_anchor.occupations,[alpha,beta]):
                raise ValueError('the frozen ROKS anchor disagrees with the selected promotion')
            result=evaluate_anchored_roks(calc,coordinates,frozen_anchor,**options)
    else:
        options.update(density_tolerance=1e-10, energy_tolerance=1e-12, maximum_iterations=1000)
        result = evaluate_gfn2_deltascf(calc, coordinates, alpha, beta, **options)
    selected = result.to_apoc(backend.electronic_state)
    overlap = anchor_overlap = None
    overlap_function = singlet_csf_overlap if mode == 'roks' else determinant_overlap
    for accepted, label in ((component.reference, 'reference'), (component.anchor, 'anchor')):
        if accepted is not None:
            previous, previous_apoc = accepted
            cross = component._cross_overlap(previous.reference_state, state)
            value = overlap_function(previous_apoc, selected, cross)
            value = value[0] if isinstance(value, tuple) else value
            if label == 'reference':
                overlap = float(value)
            else:
                anchor_overlap = float(value)
    hessian = None
    if compute_hessian:
        if frozen_anchor is None:
            hessian = evaluate_gfn2_deltascf_analytic_hessian(calc, coordinates, alpha, beta,
                mode=mode, reference_result=result,spin_reference=response_reference).hessian_hartree_per_bohr2
        else:
            from matrix_gfn2 import evaluate_anchored_roks_analytic_hessian
            hessian=evaluate_anchored_roks_analytic_hessian(calc,coordinates,frozen_anchor,
                reference_result=result).hessian_hartree_per_bohr2
    return PointEvaluationResult(0, 0., energy_hartree=result.energy_hartree,
        gradient_hartree_per_bohr=result.gradient_hartree_per_bohr.reshape(-1),
        hessian_hartree_per_bohr2=hessian, backend_coordinates_angstrom=np.asarray(coordinates),
        scf_state=(result, selected), source=f'experimental native {mode} QM/QM component',
        execution={'response_model': mode, 'stationary_reference': diagnostics,
            'selected_electronic_state': backend.electronic_state,
            'gradient_electronic_state': backend.electronic_state,
            'state_assignment_overlap': overlap, 'state_anchor_overlap': anchor_overlap,
            'state_following_contract': 'apoc_cross_ao_csf' if mode == 'roks' else 'apoc_cross_ao_determinant',
            'spin_squared': result.spin_squared, 'orbital_residual': result.orbital_residual,
            'spin_pure_singlet': mode == 'roks',
            'no_offset_response_fingerprint': None if response_reference is None else response_reference.fingerprint,
            'frozen_roks_anchor_fingerprint': None if frozen_anchor is None else frozen_anchor.fingerprint,
            'orbital_saddle_index': getattr(result,'orbital_saddle_index',None),
            'target_orbital_saddle_index': backend.orbital_saddle_index,
            'hessian_source': 'analytic' if compute_hessian else None})
