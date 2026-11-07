"""Fixed 50/50 mean of independently stationary ROKS energy surfaces.

Total excited-state energies, forces and relaxed nuclear Hessians are averaged
AFTER separate electronic optimizations. This is not ROKS of a mixed functional
and has no single wavefunction or single orbital saddle index. Excitation energy
subtracts the independently optimized ground energy of each component.
"""
from dataclasses import dataclass
import numpy as np
from .roks import ROKSSingletResult,evaluate_gfn2_roks_singlet
from .deltascf_analytic_hessian import evaluate_gfn2_deltascf_analytic_hessian


class CompositeROKSStateMismatch(ValueError):
    """Individually stationary components no longer satisfy the CSF guard."""


@dataclass(frozen=True)
class CompositeROKSResult:
    components: tuple
    component_csf_overlap: float
    component_active_pair_overlap: float | None = None
    component_tracking: str = "csf"

    @property
    def energy_hartree(self):return .5*sum(r.energy_hartree for r in self.components)
    @property
    def excitation_energy_hartree(self):
        values=[r.excitation_energy_hartree for r in self.components]
        return None if any(v is None for v in values) else .5*sum(values)
    @property
    def gradient_hartree_per_bohr(self):
        values=[r.gradient_hartree_per_bohr for r in self.components]
        return None if any(v is None for v in values) else .5*(values[0]+values[1])
    @property
    def spin_squared(self):return 0.
    @property
    def orbital_saddle_indices(self):return tuple(r.orbital_saddle_index for r in self.components)


def combine_roks_results(raw,spectral,*,minimum_component_overlap=.7,component_tracking="csf"):
    if component_tracking not in {'csf','active_pair'}:raise ValueError('unknown composite state comparison')
    if not 0<=minimum_component_overlap<=1:raise ValueError('invalid component overlap threshold')
    if not all(isinstance(r,ROKSSingletResult) and r.converged and r.orbital_residual<=1e-8 for r in (raw,spectral)):
        raise ValueError('both 50/50 components must be stationary ROKS singlets')
    if (not np.array_equal(raw.occupations,spectral.occupations)
        or not np.array_equal(raw.reference_state.atomic_numbers,spectral.reference_state.atomic_numbers)
        or not np.allclose(raw.reference_state.coordinates_bohr,spectral.reference_state.coordinates_bohr,atol=1e-10,rtol=0)
        or not np.allclose(raw.overlap,spectral.overlap,atol=1e-10,rtol=0)):
        raise ValueError('50/50 components have different states, atoms or geometries')
    if not np.array_equal(raw.core_electrons,spectral.core_electrons):
        raise ValueError('50/50 components have different frozen cores')
    # Across these explicitly different Hamiltonians compare the actual CSFs.
    # APOC's trajectory guard correctly rejects a change of Hamiltonian, so it
    # must not be disabled globally to perform this separate diagnostic.
    amplitude=0.
    for left in raw.to_apoc().components:
        for right in spectral.to_apoc().components:
            value=.5
            for ca,oa,cb,ob in zip(left.wavefunction.mo_coefficients,left.wavefunction.mo_occupations,
                                  right.wavefunction.mo_coefficients,right.wavefunction.mo_occupations):
                value*=np.linalg.det(ca[:,oa>.5].T@raw.overlap@cb[:,ob>.5])
            amplitude+=value
    score=float(abs(amplitude))
    active=None
    guarded=score
    if component_tracking=='active_pair':
        from .roks_tracking import active_orbital_character
        if not raw.spin_reference_fingerprint or not spectral.spin_reference_fingerprint:
            raise ValueError('active-pair comparison requires explicit frozen response identities')
        active=active_orbital_character(raw,spectral,raw.reference_state.coordinates_bohr,allow_different_response=True)['transition']
        guarded=active
    if guarded<minimum_component_overlap:raise CompositeROKSStateMismatch(f'50/50 component {component_tracking} overlap below threshold: overlap={guarded:.6f}')
    return CompositeROKSResult((raw,spectral),score,active,component_tracking)


def evaluate_composite_roks(calculators,coordinates_angstrom,alpha_occupations,beta_occupations,
                            response_references,*,reference_result=None,gradient=False,
                            component_options=None,minimum_component_overlap=.7):
    if len(calculators)!=2 or len(response_references)!=2:raise ValueError('50/50 requires exactly two ordered components: raw, spectral')
    if reference_result is not None and not isinstance(reference_result,CompositeROKSResult):raise TypeError('expected a composite retained state')
    options=({}, {}) if component_options is None else component_options
    if len(options)!=2:raise ValueError('provide solver options for each component')
    results=[]
    for i in range(2):
        results.append(evaluate_gfn2_roks_singlet(calculators[i],coordinates_angstrom,alpha_occupations,beta_occupations,
            spin_reference=response_references[i],reference_result=None if reference_result is None else reference_result.components[i],
            gradient=gradient,**options[i]))
    return combine_roks_results(*results,minimum_component_overlap=minimum_component_overlap)


@dataclass(frozen=True)
class CompositeROKSHessian:
    reference: CompositeROKSResult
    components: tuple
    @property
    def hessian_hartree_per_bohr2(self):return .5*sum(h.hessian_hartree_per_bohr2 for h in self.components)
    method: str = 'mean_of_two_independently_relaxed_analytic_nuclear_hessians'


def evaluate_composite_roks_analytic_hessian(calculators,coordinates_angstrom,response_references,*,reference_result,**options):
    if len(calculators)!=2 or len(response_references)!=2:raise ValueError('50/50 requires two ordered components')
    if not isinstance(reference_result,CompositeROKSResult):raise TypeError('expected composite stationary state')
    checked=combine_roks_results(*reference_result.components,component_tracking=reference_result.component_tracking)
    components=[]
    for calc,ref,result in zip(calculators,response_references,checked.components):
        components.append(evaluate_gfn2_deltascf_analytic_hessian(calc,coordinates_angstrom,*result.occupations,
            mode='roks',spin_reference=ref,reference_result=result,**options))
    return CompositeROKSHessian(checked,tuple(components))
