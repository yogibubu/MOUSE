"""Research surface: native ground energy plus the fixed 50/50 ROKS gaps.

E(R) = E0_native(R) + 1/2 sum_i [E_ROKS_i(R) - E0_i(R)].
The ground replacement varies with geometry and includes ALL its derivatives.
It is not a fitted constant or a redefinition of the two response models.
"""
from dataclasses import dataclass
import numpy as np
from .composite_roks import CompositeROKSResult, evaluate_composite_roks_analytic_hessian
from .restricted_ground import (RestrictedGroundResult, evaluate_restricted_ground,
                                restricted_ground_analytic_hessian)
from .shared_pair_roks import SharedPairROKSResult,evaluate_shared_pair_roks_analytic_hessian

from .relative_pair_roks import RelativePairROKSResult,evaluate_relative_pair_roks_analytic_hessian


@dataclass(frozen=True)
class GroundReferencedROKSResult:
    excited: CompositeROKSResult | SharedPairROKSResult | RelativePairROKSResult
    grounds: tuple

    def __post_init__(self):
        if not isinstance(self.excited, (CompositeROKSResult,SharedPairROKSResult,RelativePairROKSResult)) or len(self.grounds) != 3:
            raise ValueError('ground-referenced 50/50 needs two ROKS and three ground solutions')
        template = self.excited.components[0].reference_state
        fingerprints = [None]+[r.spin_reference_fingerprint for r in self.excited.components]
        for r, fingerprint in zip(self.grounds, fingerprints):
            if (not isinstance(r, RestrictedGroundResult) or not r.converged
                    or r.orbital_residual > 1e-8
                    or not np.isfinite(r.minimum_orbital_curvature_hartree)
                    or r.minimum_orbital_curvature_hartree <= 0
                    or r.orbital_saddle_index != 0 or r.spin_reference_fingerprint != fingerprint
                    or not np.array_equal(r.occupations[0], r.occupations[1])
                    or not np.allclose(r.coefficients[0], r.coefficients[1], atol=1e-10, rtol=0)
                    or np.sum(r.occupations) != np.sum(self.excited.components[0].occupations)
                    or not np.array_equal(r.reference_state.atomic_numbers, template.atomic_numbers)
                    or r.reference_state.model_id != template.model_id
                    or r.reference_state.charge != template.charge
                    or not np.allclose(r.reference_state.coordinates_bohr, template.coordinates_bohr,
                                       atol=1e-10, rtol=0)):
                raise ValueError('ground replacement has unmatched or nonstationary components')

    def __getattr__(self, name):
        return getattr(self.excited, name)

    @property
    def energy_hartree(self):
        return self.excited.energy_hartree+self.grounds[0].energy_hartree-.5*sum(
            r.energy_hartree for r in self.grounds[1:])

    @property
    def excitation_energy_hartree(self):
        return self.energy_hartree-self.grounds[0].energy_hartree

    @property
    def gradient_hartree_per_bohr(self):
        gradients = [self.excited.gradient_hartree_per_bohr]+[
            r.gradient_hartree_per_bohr for r in self.grounds]
        if any(g is None for g in gradients):
            return None
        return gradients[0]+gradients[1]-.5*(gradients[2]+gradients[3])

    @property
    def component_gaps_hartree(self):
        return tuple(e.energy_hartree-g.energy_hartree for e, g in
                     zip(self.excited.components, self.grounds[1:]))

    @property
    def spectrum_qualified(self):
        return all(gap > 0 for gap in self.component_gaps_hartree)


def reference_roks_to_native_ground(calculator, coordinates_angstrom, responses, *,
                                    excited, reference_result=None, gradient=True):
    """Solve the three restricted references without rerunning the excited CSFs."""
    if len(responses) != 2:
        raise ValueError('provide ordered full and spectral response models')
    if reference_result is not None and not isinstance(reference_result, GroundReferencedROKSResult):
        raise TypeError('ground warm start requires a ground-referenced ROKS result')
    template = excited.components[0].reference_state
    grounds = []
    for i, response in enumerate([None]+list(responses)):
        grounds.append(evaluate_restricted_ground(calculator, coordinates_angstrom,
            template=template, spin_reference=response, gradient=gradient,
            reference_result=None if reference_result is None else reference_result.grounds[i]))
    return GroundReferencedROKSResult(excited, tuple(grounds))


@dataclass(frozen=True)
class GroundReferencedROKSHessian:
    reference: GroundReferencedROKSResult
    excited: object
    grounds: tuple
    method: str = 'analytic_native_ground_plus_mean_roks_gaps'

    @property
    def hessian_hartree_per_bohr2(self):
        return self.excited.hessian_hartree_per_bohr2+self.grounds[0].hessian_hartree_per_bohr2-.5*sum(
            r.hessian_hartree_per_bohr2 for r in self.grounds[1:])


def ground_referenced_roks_analytic_hessian(calculator, coordinates_angstrom, responses, *,
                                           reference_result):
    if not isinstance(reference_result, GroundReferencedROKSResult):
        raise TypeError('expected a ground-referenced ROKS result')
    if isinstance(reference_result.excited,RelativePairROKSResult):
        excited=evaluate_relative_pair_roks_analytic_hessian(calculator,coordinates_angstrom,
            responses,reference_result=reference_result.excited)
    elif isinstance(reference_result.excited,SharedPairROKSResult):
        excited=evaluate_shared_pair_roks_analytic_hessian(calculator,coordinates_angstrom,
            responses,reference_result=reference_result.excited)
    else:
        excited = evaluate_composite_roks_analytic_hessian((calculator, calculator),
            coordinates_angstrom, responses, reference_result=reference_result.excited)
    grounds = tuple(restricted_ground_analytic_hessian(calculator, coordinates_angstrom,
        reference_result=r, spin_reference=response) for r, response in
        zip(reference_result.grounds, [None]+list(responses)))
    return GroundReferencedROKSHessian(reference_result, excited, grounds)
