"""Restricted ground-state direct optimization, independent of SCC iteration.

This auxiliary solver also applies to the frozen response Hamiltonians. It
returns an ordinary closed-shell determinant, never an excited ROKS CSF.
"""
from dataclasses import dataclass, replace
from types import SimpleNamespace
import numpy as np
from scipy.linalg import expm
from .deltascf import DeltaSCFResult, prepare_gfn2_deltascf, project_delta_orbitals
from .orbital_saddle import generalized_mode_following
from .roks_orbital_response import orbital_residual_jacobian


@dataclass(frozen=True)
class RestrictedGroundResult:
    reference: DeltaSCFResult
    minimum_orbital_curvature_hartree: float
    orbital_saddle_index: int = 0

    def __getattr__(self, name):
        return getattr(self.reference, name)


def evaluate_restricted_ground(calculator, coordinates_angstrom, *, template,
                              spin_reference=None, reference_result=None,
                              gradient=False, orbital_tolerance=1e-10,
                              maximum_iterations=250):
    """Minimize all shared occupied/virtual rotations with integer occupation.

    The template supplies AO metadata only. Every energy is the absolute
    physical energy of the selected Hamiltonian; no SCC energy is substituted.
    Success requires a stationary restricted minimum, including its curvature.
    """
    context = prepare_gfn2_deltascf(calculator, coordinates_angstrom,
        orbital_reference=template, spin_reference=spin_reference)
    state = context.state
    if state.multiplicity != 1 or state.electron_count % 2:
        raise ValueError('restricted ground solver requires an even-electron singlet')
    occ = np.array([state.alpha_occupations, state.beta_occupations])
    c0 = state.coefficients
    if reference_result is not None:
        raw=reference_result.reference if isinstance(reference_result,RestrictedGroundResult) else reference_result
        if (not isinstance(raw,DeltaSCFResult) or not np.array_equal(raw.occupations[0],raw.occupations[1])
                or not np.allclose(raw.coefficients[0],raw.coefficients[1],atol=1e-10,rtol=0)):
            raise TypeError('restricted warm start needs a closed-shell determinant')
        if raw.spin_reference_fingerprint != (None if spin_reference is None else spin_reference.fingerprint):
            raise ValueError('restricted seed has a different response Hamiltonian')
        c0 = project_delta_orbitals(raw, context)[0]
    left, right = np.triu_indices(len(c0), 1)
    active = occ[0, left] != occ[0, right]
    left, right = left[active], right[active]
    cache = {}

    def objective(c, curvature=True):
        if 'c' not in cache or not np.array_equal(c, cache['c']):
            density = np.array([(c*o)@c.T for o in occ])
            f, energy = context.fock_and_energy(density)
            q = sum((c.T@ff@c)*o[None, :] for ff, o in zip(f, occ))
            cache.clear()
            cache.update(c=c.copy(), f=f, e=energy,
                         g=2*(q[left, right]-q[right, left]))
        if curvature and 'h' not in cache:
            derivative_state = SimpleNamespace(**vars(state))
            derivative_state.coefficients = c
            kernel = context.density_response_kernel(derivative_state)
            h = orbital_residual_jacobian(c, occ, cache['f'], kernel,
                                          context.polarization, left, right)
            cache['h'] = .5*(h+h.T)
        return cache['e'], cache['g'], cache.get('h')

    def retract(c, step):
        k = np.zeros_like(c)
        k[left, right] = step
        k[right, left] = -step
        return c@expm(k)

    solved = generalized_mode_following(objective, retract, c0, saddle_index=0,
        gradient_tolerance=orbital_tolerance, maximum_iterations=maximum_iterations)
    if not solved.converged:
        raise RuntimeError(f'restricted ground direct optimization failed: {solved.history[-1]}')
    c = solved.point
    energy, residual, _ = objective(c)
    result = DeltaSCFResult(np.array([c, c]), occ, state.overlap, cache['f'],
        float(energy), solved.iterations, 0., float(np.max(abs(residual))), True,
        np.array([c0, c0]), reference_state=state,
        core_electrons=np.asarray(state.atomic_numbers)-np.asarray(calculator._valence),
        spin_reference_fingerprint=None if spin_reference is None else spin_reference.fingerprint)
    if gradient:
        q = sum((c.T@ff@c)*o[None, :] for ff, o in zip(cache['f'], occ))
        weighted = c@(.5*(q+q.T))@c.T
        result = replace(result, gradient_hartree_per_bohr=context.analytic_gradient(
            result.spin_densities_ao, weighted))
    return RestrictedGroundResult(result, float(np.min(solved.eigenvalues)))


def restricted_ground_analytic_hessian(calculator, coordinates_angstrom, *,
                                     reference_result, spin_reference=None):
    """Relaxed analytic Hessian on the restricted common-orbital manifold."""
    from .deltascf_analytic_hessian import (_fixed_density_geometry,
        _response_blocks, _add_charge_response, AnalyticDeltaSCFHessian)
    from .calculator import ANGSTROM_TO_BOHR
    from .spin_reference import validate_spin_reference
    if not isinstance(reference_result, RestrictedGroundResult):
        raise TypeError('expected a certified restricted stationary determinant')
    r = reference_result.reference
    validate_spin_reference(r, spin_reference)
    if (not r.converged or r.orbital_residual > 1e-8 or not np.allclose(
            r.reference_state.coordinates_bohr, np.asarray(coordinates_angstrom)*ANGSTROM_TO_BOHR,
            atol=1e-10, rtol=0)):
        raise ValueError('restricted Hessian needs a matching stationary geometry')
    context = prepare_gfn2_deltascf(calculator, coordinates_angstrom,
        orbital_reference=r.reference_state, spin_reference=spin_reference)
    kernel, ds, df, d2s, err = _fixed_density_geometry(context, r.spin_densities_ao)
    if context.charge_correction is not None:
        charge = context.charge_correction
        dfield, efield = charge.geometry(r.density_ao, ds, d2s)
        df, err = df+dfield, err+efield
        kernel = _add_charge_response(kernel,charge)
    rr, kk, kr = _response_blocks([r.coefficients[0]],
        [(1., r.occupations, r.fock)], ds, d2s, [(df, err)], [kernel], context.polarization)
    kk = .5*(kk+kk.T)
    minimum = float(np.min(np.linalg.eigvalsh(kk)))
    if minimum < 1e-8:
        raise ValueError('restricted ground orbital response is singular or unstable')
    response = np.linalg.solve(kk, -kr)
    correction = kr.T@response
    total = rr+correction
    return AnalyticDeltaSCFHessian(reference_result, .5*(total+total.T), rr,
        correction, kk, kr, response, float(np.max(abs(kk@response+kr))),
        float(np.max(abs(total-total.T))), minimum,
        method='analytic_restricted_ground_schur_complement')
