"""State-preserving initialization between two fixed one-body ROKS models.

Intermediate operators guide the solver only. The returned stationary point
belongs to the exact requested endpoint, with no penalty or energy correction.
"""
from dataclasses import dataclass, fields, replace
import numpy as np
from .nonlocal_response import FrozenNonlocalResponseReference
from .roks import ROKSSingletResult, evaluate_gfn2_roks_singlet
from .roks_tracking import make_roks_trial_guard
from .deltascf import prepare_gfn2_deltascf


@dataclass(frozen=True)
class ROKSContinuationState:
    fraction: float
    result: ROKSSingletResult
    origin_fingerprint: str
    target_fingerprint: str
    history: tuple


class ROKSStateFollowingError(RuntimeError):
    def __init__(self, message, accepted):
        self.accepted = accepted
        super().__init__(message)


def continue_roks_one_body_response(calculator, coordinates_angstrom, responses, *,
        reference_result, reference_coefficients, occupied, virtual,
        reference_transition, projected_reference_norm,
        restart=None, checkpoint=None, initial_step=.2, minimum_step=.0005,
        maximum_step=.2, orbital_tolerance=1e-9, maximum_evaluations=80,
        orbital_step=.08, minimum_local_overlap=.90, minimum_active_overlap=.70,
        minimum_transition_fidelity=.90, minimum_sector_norm=.70):
    """Continue an assigned singlet to the second model with analytic Newton.

    Models must have identical charge/spin response, basis, geometry and frame
    definitions and differ only in their one-body operator and source label.
    Reference transition directions and their projection norm are supplied by
    the caller; no excitation reference energies enter the calculation.

    Each trial keeps local full-CSF continuity and fixed active/transition
    character. The index is certified from the full analytic orbital curvature,
    rather than restricted to a guessed list. Checkpoints expose accepted
    fractions only; rejected trials never become restart seeds.
    """
    if (len(responses) != 2 or not isinstance(reference_result, ROKSSingletResult)
            or not reference_result.converged):
        raise ValueError('continuation requires two models and a stationary assigned ROKS seed')
    first, last = responses
    if (not isinstance(first, FrozenNonlocalResponseReference) or type(first) is not type(last)):
        raise TypeError('continuation requires matching full AO-pair response models')
    for field in fields(first):
        if field.name in {'source_model', 'pair_linear_hartree'}:
            continue
        a, b = getattr(first, field.name), getattr(last, field.name)
        if not np.array_equal(a, b):
            raise ValueError('one-body continuation cannot change '+field.name)
    if (not np.isfinite(projected_reference_norm) or projected_reference_norm < .7
            or projected_reference_norm > 1+1e-5):
        raise ValueError('assigned reference is not representable in the compact occupied/virtual spaces')
    if (not 0 < minimum_step <= initial_step <= maximum_step <= 1
            or maximum_evaluations < 1 or not np.isfinite(orbital_tolerance)
            or orbital_tolerance <= 0 or not np.isfinite(orbital_step) or orbital_step <= 0):
        raise ValueError('invalid continuation controls')
    template = reference_result.reference_state
    if template is None:
        raise ValueError('continuation needs a molecular reference')
    # Check that the start model really is the Hamiltonian of the supplied
    # stationary seed, before replacing its provenance/transport metadata.
    context = prepare_gfn2_deltascf(calculator, coordinates_angstrom,
                                  orbital_reference=template, spin_reference=first)
    fm, em = context.fock_and_energy(reference_result.densities())
    ft, et = context.fock_and_energy(reference_result.densities(triplet=True))
    if (abs(2*em-et-reference_result.energy_hartree) > 1e-9
            or np.max(abs(fm-reference_result.fock_mixed)) > 1e-8
            or np.max(abs(ft-reference_result.fock_triplet)) > 1e-8):
        raise ValueError('continuation seed does not belong to the starting Hamiltonian')
    seed = replace(reference_result, spin_reference_fingerprint=first.fingerprint,
                   spin_frame_clusters=getattr(first, 'frame_clusters', None))
    accepted = ROKSContinuationState(0., seed, first.fingerprint, last.fingerprint, ())
    if restart is not None:
        if (not isinstance(restart, ROKSContinuationState)
                or restart.origin_fingerprint != first.fingerprint
                or restart.target_fingerprint != last.fingerprint
                or not 0 <= restart.fraction <= 1 or not restart.result.converged):
            raise ValueError('continuation restart belongs to another problem')
        accepted = restart
    def model(t):
        if t == 0:
            return first
        if t == 1:
            return last
        return replace(first, source_model='ROKS/continuation/'+first.fingerprint+'/'+last.fingerprint+'/'+float(t).hex(),
                       pair_linear_hartree=(1-t)*first.pair_linear_hartree+t*last.pair_linear_hartree)
    expected = model(accepted.fraction).fingerprint
    if accepted.result.spin_reference_fingerprint != expected:
        raise ValueError('continuation restart has inconsistent endpoint provenance')
    step = initial_step
    while accepted.fraction < 1:
        fraction = min(1., accepted.fraction+step)
        target = model(fraction)
        current = accepted.result
        guard = make_roks_trial_guard(context.state.overlap, current.occupations,
            seed.common_coefficients, minimum_local_overlap=minimum_local_overlap,
            minimum_active_overlap=minimum_active_overlap,
            reference_coefficients=reference_coefficients, occupied=occupied, virtual=virtual,
            reference_transition=reference_transition, transition_metric='rank_one',
            minimum_transition_overlap=minimum_transition_fidelity,
            minimum_sector_norm=minimum_sector_norm)
        if not guard(current.common_coefficients, current.common_coefficients):
            raise ROKSStateFollowingError('accepted seed fails fixed transition/active gates', accepted)
        trial_seed = replace(current, spin_reference_fingerprint=target.fingerprint,
            spin_frame_clusters=getattr(target, 'frame_clusters', None), converged=False,
            orbital_saddle_index=None, target_orbital_saddle_index=None)
        try:
            result = evaluate_gfn2_roks_singlet(calculator, coordinates_angstrom,
                *current.occupations, reference_result=trial_seed, spin_reference=target,
                reference_policy='geometry_only', minimum_state_overlap=minimum_local_overlap,
                orbital_solver='newton', orbital_tolerance=orbital_tolerance,
                maximum_evaluations=maximum_evaluations, maximum_step=orbital_step,
                certify_index=True, trial_guard=guard)
            if not guard(current.common_coefficients, result.common_coefficients):
                raise ValueError('stationary endpoint lost assigned character')
        except (RuntimeError, ValueError) as exc:
            row = dict(from_fraction=accepted.fraction, to_fraction=fraction,
                       accepted=False, reason=str(exc), guard=guard.last_scores)
            accepted = replace(accepted, history=accepted.history+(row,))
            step *= .5
            if step < minimum_step:
                raise ROKSStateFollowingError('cannot continue the assigned branch to the endpoint', accepted) from exc
            continue
        row = dict(from_fraction=accepted.fraction, to_fraction=fraction, accepted=True,
                   index=result.orbital_saddle_index, residual=result.orbital_residual,
                   guard=guard.last_scores, guard_trials=guard.evaluations,
                   guard_rejections=int(guard.rejections))
        accepted = ROKSContinuationState(fraction, result, first.fingerprint, last.fingerprint,
                                         accepted.history+(row,))
        if checkpoint is not None:
            checkpoint(accepted)
        step = min(maximum_step, step*1.5)
    return accepted
