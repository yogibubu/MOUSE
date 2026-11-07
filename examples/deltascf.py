"""Experimental broken-symmetry DeltaSCF example; run on Oracle/MAC-STUDIO."""
import numpy as np
from matrix_gfn2 import (GFN2Calculator, evaluate_gfn2_deltascf,
                         spin_adapt_gfn2_singlet, evaluate_gfn2_deltascf_analytic_hessian)
from matrix_gfn2.response_reference import stationary_response_reference

atoms = ['C', 'O', 'H', 'H']
xyz = np.array([[0., 0., 0.], [1.21, 0., 0.], [-.58, .94, 0.], [-.58, -.94, 0.]])
calculator = GFN2Calculator(atoms, temperature_kelvin=0.)
ground, certificate = stationary_response_reference(calculator, xyz)
alpha = ground.orbital_state.alpha_occupations.copy()
beta = ground.orbital_state.beta_occupations.copy()
# Illustrative HOMO->LUMO promotion only; assign orbital character for real use.
homo = np.flatnonzero(alpha > .5)[-1]
lumo = np.flatnonzero(alpha < .5)[0]
alpha[homo], alpha[lumo] = 0., 1.
result = evaluate_gfn2_deltascf(calculator, xyz, alpha, beta, ground_state=ground,
    gradient=True, density_tolerance=1e-10, orbital_tolerance=1e-10,
    energy_tolerance=1e-12, maximum_iterations=1000)
print('Spin-polarized DeltaSCF gap / hartree:', result.excitation_energy_hartree)
print('Determinant <S^2> (no spin purification):', result.spin_squared)
print('APOC state:', result.to_apoc().state_id)
result.save_checkpoint('deltascf_checkpoint.npz')

singlet = spin_adapt_gfn2_singlet(calculator, result)
analytic = evaluate_gfn2_deltascf_analytic_hessian(
    calculator, xyz, alpha, beta, mode='roks', reference_result=singlet)
print('Reoptimized singlet CSF <S^2>:', singlet.spin_squared)
print('Analytic singlet Hessian / hartree per bohr^2:', analytic.hessian_hartree_per_bohr2)
