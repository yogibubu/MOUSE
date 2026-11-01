"""Experimental spin-polarized integer-occupation Delta-SCF for GFN2.

IMOM relaxes separate alpha/beta orbitals using an on-site shell spin energy.
Broken-symmetry Ms=0 determinants are not spin-pure singlet eigenstates.
Analytic nuclear gradients and a coupled-response analytic Hessian are exposed
through the derivative modules. GMF and freeze-and-release are not implemented.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import eigh


@dataclass(frozen=True)
class DeltaSCFResult:
    coefficients: np.ndarray
    occupations: np.ndarray
    overlap: np.ndarray
    fock: np.ndarray
    energy_hartree: float
    iterations: int
    density_residual: float
    orbital_residual: float
    converged: bool
    reference_coefficients: np.ndarray
    ground_energy_hartree: float | None = None
    reference_state: object | None = None
    core_electrons: np.ndarray | None = None
    gradient_hartree_per_bohr: np.ndarray | None = None
    spin_reference_fingerprint: str | None = None

    @property
    def density_ao(self):
        return np.sum(self.spin_densities_ao, axis=0)

    @property
    def spin_densities_ao(self):
        return np.array([(c * o) @ c.T for c, o in zip(self.coefficients, self.occupations, strict=True)])

    @property
    def spin_squared(self):
        na, nb = np.sum(self.occupations, axis=1)
        a = self.coefficients[0][:, self.occupations[0] > .5]
        b = self.coefficients[1][:, self.occupations[1] > .5]
        return float(((na-nb)/2)**2 + (na+nb)/2 - np.sum(np.abs(a.T @ self.overlap @ b)**2))

    @property
    def excitation_energy_hartree(self):
        return None if self.ground_energy_hartree is None else self.energy_hartree - self.ground_energy_hartree

    def to_apoc(self, state_id=1):
        """Export physical nuclei and explicit frozen cores for determinant tracking."""
        from matrix_apoc import DeterminantOrbitals, DeltaSCFState
        from .spin_polarization import SPIN_PARAMETER_SHA256
        state = self.reference_state
        if state is None or self.core_electrons is None:
            raise ValueError("APOC export requires molecular metadata from evaluate_gfn2_deltascf")
        counts = np.sum(self.occupations, axis=1)
        return DeltaSCFState(state_id, DeterminantOrbitals(
            state.atomic_numbers, state.coordinates_bohr, self.overlap,
            tuple(self.coefficients), tuple(self.occupations), self.core_electrons,
            charge=state.charge, multiplicity=int(round(counts[0]-counts[1]))+1,
            total_energy_hartree=self.energy_hartree,
            model=state.model_id+".spin."+(self.spin_reference_fingerprint or SPIN_PARAMETER_SHA256)))

    def save_checkpoint(self, path):
        """Persist the trial orbitals AND original IMOM reference for restart."""
        with Path(path).open('wb') as stream:
            np.savez_compressed(stream, coefficients=self.coefficients,
                                occupations=self.occupations, overlap=self.overlap,
                                reference_coefficients=self.reference_coefficients,
                                energy_hartree=self.energy_hartree, iterations=self.iterations,
                                converged=self.converged)


class DeltaSCFConvergenceError(RuntimeError):
    def __init__(self, result):
        self.result = result
        super().__init__(f"Delta-SCC did not converge after {result.iterations} iterations; "
                         f"density residual={result.density_residual:.3e}, "
                         f"orbital residual={result.orbital_residual:.3e}")


def run_deltascf(overlap, reference_coefficients, occupations, fock_and_energy, *,
                  initial_coefficients=None, maximum_iterations=200, mixing=.3,
                  density_tolerance=1.e-7, orbital_tolerance=1.e-7,
                  energy_tolerance=1.e-9, strict=True, selection="imom"):
    """IMOM SCF using the complete F[P], E[P] callback in one fixed AO basis.

    The callback accepts (2, nao, nao) spin densities and returns two Focks.
    Two explicit spin channels are required. Selection uses projection onto
    each original occupied subspace, invariant to its internal rotations.
    Nonconvergence raises with the resumable result in ``exception.result``.
    """
    s = np.asarray(overlap, dtype=float)
    c0 = np.asarray(reference_coefficients, dtype=float)
    occ = np.asarray(occupations, dtype=float)
    if (s.ndim != 2 or s.shape[0] != s.shape[1] or not np.all(np.isfinite(s))
            or not np.allclose(s, s.T, atol=1.e-12, rtol=0)
            or c0.shape != (2, len(s), len(s)) or occ.shape != (2, len(s))
            or not np.all(np.isfinite(c0)) or not np.all(np.isfinite(occ))
            or not np.all((occ == 0) | (occ == 1))
            or selection not in {"imom", "aufbau"}
            or maximum_iterations < 1 or not 0 < mixing <= 1
            or min(density_tolerance, orbital_tolerance, energy_tolerance) <= 0):
        raise ValueError("invalid Delta-SCC orbitals, metric, occupations or convergence settings")
    np.linalg.cholesky(s)
    c = c0.copy() if initial_coefficients is None else np.array(initial_coefficients, dtype=float, copy=True)
    for orbitals in (c0, c):
        if (orbitals.shape != c0.shape or not np.all(np.isfinite(orbitals))
                or not all(np.allclose(x.T @ s @ x, np.eye(len(s)), atol=2.e-7, rtol=0) for x in orbitals)):
            raise ValueError("Delta-SCC orbitals must be complete and AO-metric orthonormal")
    refs = [x[:, o > .5] for x, o in zip(c0, occ, strict=True)]
    def density(orbitals):
        return np.array([(x * o) @ x.T for x, o in zip(orbitals, occ, strict=True)])
    def evaluate(p):
        f, e = fock_and_energy(p)
        f = np.asarray(f, dtype=float)
        if (f.shape != (2, *s.shape) or not np.all(np.isfinite(f)) or not np.isfinite(e)
                or not np.allclose(f, f.transpose(0, 2, 1), atol=1.e-10, rtol=0)):
            raise ValueError("Delta-SCC callback returned invalid Fock/energy")
        return f, float(e)
    p = density(c)
    old_energy = None
    converged = False
    for iteration in range(1, maximum_iterations + 1):
        f, _ = evaluate(p)
        for channel, (ref, o) in enumerate(zip(refs, occ, strict=True)):
            _, eigenvectors = eigh(f[channel], s)
            weights = np.sum(np.abs(ref.T @ s @ eigenvectors)**2, axis=0)
            nocc = ref.shape[1]
            selected = (np.argsort(-weights, kind='stable')[:nocc]
                        if selection == 'imom' else np.arange(nocc))
            virtual = np.setdiff1d(np.arange(len(s)), selected)
            c[channel][:, o > .5] = eigenvectors[:, selected]
            c[channel][:, o < .5] = eigenvectors[:, virtual]
        output = density(c)
        residual = float(np.max(np.abs(output - p)))
        # Certify the determinant actually returned, not its mixed precursor.
        final_fock, final_energy = evaluate(output)
        orbital_residual = max(float(np.max(np.abs(fspin @ d @ s - s @ d @ fspin)))
                               for fspin, d in zip(final_fock, output, strict=True))
        if (old_energy is not None and abs(final_energy - old_energy) < energy_tolerance
                and residual < density_tolerance and orbital_residual < orbital_tolerance):
            converged = True
            break
        p = (1 - mixing) * p + mixing * output
        old_energy = final_energy
    result = DeltaSCFResult(c.copy(), occ.copy(), s.copy(), final_fock,
                           final_energy, iteration, residual, orbital_residual,
                           converged, c0.copy())
    if strict and not converged:
        raise DeltaSCFConvergenceError(result)
    return result


def evaluate_gfn2_deltascf(calculator, coordinates_angstrom, alpha_occupations,
                            beta_occupations, *, ground_state=None, gradient=False,
                            reference_result=None, minimum_state_overlap=.70,
                            spin_reference=None,**solver_options):
    """Relax a promoted determinant using resident GFN2 plus shell spin energy.

    ``ground_state`` may reuse an existing retained result at exactly this
    geometry. A zero-temperature standard GFN2 calculator is required. The
    immutable model's charge/multipole/D4 terms are recomputed at every trial
    density; geometry-only terms cancel in the energy difference.
    """
    from dataclasses import replace
    from .spin_reference import validate_spin_reference
    if reference_result is not None:
        validate_spin_reference(reference_result,spin_reference)
    context = prepare_gfn2_deltascf(calculator, coordinates_angstrom, ground_state=ground_state,
                                  spin_reference=spin_reference)
    state, callback = context.state, context.fock_and_energy
    occ = np.asarray([alpha_occupations, beta_occupations], dtype=float)
    if (occ.shape != (2, state.nmo)
            or not np.all(np.isfinite(occ)) or not np.all((occ == 0) | (occ == 1))
            or np.sum(occ) != state.electron_count
            or np.sum(occ[0]) < np.sum(occ[1])):
        raise ValueError("Delta-SCF occupations must conserve electron number with Nalpha >= Nbeta")
    # The excitation gap must use a reference relaxed with the SAME spin energy.
    ground_occupations = np.array([state.alpha_occupations, state.beta_occupations])
    ground_options = {key: value for key, value in solver_options.items()
                      if key not in {"initial_coefficients", "selection", "strict"}}
    polarized_ground = run_deltascf(state.overlap, np.array([state.coefficients]*2),
                                    ground_occupations, callback, selection="aufbau",
                                    strict=True, **ground_options)
    reference_coefficients = polarized_ground.coefficients
    if reference_result is not None:
        reference_coefficients = project_delta_orbitals(reference_result, context)
    result = run_deltascf(state.overlap, reference_coefficients, occ,
                           callback, **solver_options)
    result = replace(result, ground_energy_hartree=polarized_ground.energy_hartree,
                   reference_state=state,
                   core_electrons=np.asarray(state.atomic_numbers)-np.asarray(calculator._valence, dtype=int),
                   spin_reference_fingerprint=None if spin_reference is None else spin_reference.fingerprint)
    if reference_result is not None:
        from matrix_apoc import follow_deltascf_state
        follow_deltascf_state(reference_result.to_apoc(), (result.to_apoc(),),
            delta_cross_overlap(reference_result.reference_state, context),
            minimum_overlap=minimum_state_overlap)
    if gradient:
        if not result.converged:
            raise DeltaSCFConvergenceError(result)
        weighted = sum(p @ f @ p for p, f in zip(result.spin_densities_ao, result.fock, strict=True))
        g = context.analytic_gradient(result.spin_densities_ao, weighted)
        result = replace(result, gradient_hartree_per_bohr=g)
    return result


@dataclass
class DeltaSCFContext:
    calculator: object
    state: object
    kernel: object
    polarization: object
    fock_and_energy: object
    model: tuple
    charge_correction: object | None = None
    energy_only: object | None = None

    def energy(self,spin_densities):
        return self.fock_and_energy(spin_densities)[1] if self.energy_only is None else self.energy_only(spin_densities)

    def density_response_kernel(self,state):
        base=self.calculator.density_response_kernel(state)
        if self.charge_correction is None:return base
        def response(dp):return base(dp)+self.charge_correction.response(dp)
        def batch(dp):
            native=(base.response_batch(dp) if hasattr(base,'response_batch')
                    else np.asarray([base(x) for x in dp]))
            return native+self.charge_correction.response_batch(dp)
        response.response_batch=batch
        return response

    def analytic_gradient(self, spin_densities, weighted_density):
        """Exact stationary first derivative, including spin-metric and Pulay terms."""
        if getattr(self.polarization,'vertical_only',False):
            raise NotImplementedError('this response prototype is vertical-only; nuclear derivatives are unavailable')
        from .calculator import _analytic_gradient_pair_packed
        from .multipole import AtomicMultipoles
        state, kernel, calc = self.state, self.kernel, self.calculator
        basis, _, _, _, _, partition, atom_parameters, orbital_parameters, *_ = self.model
        p = np.sum(spin_densities, axis=0)
        pref = (state.coefficients*(state.alpha_occupations+state.beta_occupations)) @ state.coefficients.T
        dq, dm = kernel._observables(p-pref)
        m0 = kernel.reference_multipoles
        m = AtomicMultipoles(m0.charges+dm.charges, m0.dipoles_bohr+dm.dipoles_bohr,
                             m0.quadrupoles_bohr2+dm.quadrupoles_bohr2)
        neutral = kernel._d4(np.zeros(len(state.atomic_numbers))) if len(state.atomic_numbers) >= 3 else None
        integrals = calc._post_scf_pair_integrals
        g = _analytic_gradient_pair_packed(
            density=p, weighted_density=weighted_density, basis=basis,
            coordinates=state.coordinates_bohr, atomic_numbers=calc.atomic_numbers,
            atom_parameters=atom_parameters, orbital_parameters=orbital_parameters,
            partition=partition, shell_interaction=kernel.shell_interaction,
            shell_q=kernel.reference_shell_charges+dq, multipoles=m,
            anisotropic_parameters=kernel.anisotropic_parameters,
            anisotropic_geometry=kernel.anisotropic_geometry,
            d4=kernel._d4(m.charges), d4_parameters=kernel.d4_parameters,
            neutral_d4=neutral, valence=calc._valence,
            integrals=integrals, spherical_domains=calc._post_scf_pair_domains,
            element_values=calc._element_values, base_multipole_radii=calc._base_multipole_radii,
            valence_coordination=calc._valence_coordination,
            electronegativities=calc._electronegativities,
            orbital_metadata=calc._orbital_metadata,
            coordination_override=calc._active_coordination,
            native_context=calc._active_native_context,
        )
        # Shell constants are geometry-independent. Only Mulliken S contributes.
        _, _, magnetization = self.polarization.evaluate(spin_densities)
        v = (self.polarization.coupling_hartree @ magnetization)[self.polarization.orbital_shell]
        rows, cols = integrals.rows, integrals.columns
        weight = (spin_densities[0]-spin_densities[1])[rows, cols]
        weight *= .5*(v[rows]+v[cols])*np.where(rows == cols, 1., 2.)
        atoms = np.asarray(basis.spherical_atom)
        np.add.at(g, atoms[rows], weight[:, None]*integrals.left_derivatives[:, 0])
        np.add.at(g, atoms[cols], weight[:, None]*integrals.right_derivatives[:, 0])
        if hasattr(self.polarization,'tensor_metric_potential'):
            metric_field=self.polarization.tensor_metric_potential(spin_densities)
            factor=metric_field[rows,cols]*np.where(rows==cols,1.,2.)
            np.add.at(g,atoms[rows],factor[:,None]*integrals.left_derivatives[:,0])
            np.add.at(g,atoms[cols],factor[:,None]*integrals.right_derivatives[:,0])
        if self.charge_correction is not None:
            charge=self.charge_correction
            factor=charge.metric_potential(p)[rows,cols]*np.where(rows==cols,1.,2.)
            np.add.at(g,atoms[rows],factor[:,None]*integrals.left_derivatives[:,0])
            np.add.at(g,atoms[cols],factor[:,None]*integrals.right_derivatives[:,0])
        # Nonlocal frozen operators also move with the molecular frame.
        if hasattr(self.polarization,'explicit_gradient'):
            g=g+self.polarization.explicit_gradient(spin_densities)
        if self.charge_correction is not None and hasattr(self.charge_correction,'explicit_gradient'):
            g=g+self.charge_correction.explicit_gradient(p)
        return g


def prepare_gfn2_deltascf(calculator, coordinates_angstrom, *, ground_state=None,spin_reference=None,orbital_reference=None,allow_large=False):
    """Bind a complete, spin-polarized fixed-geometry energy/Fock functional."""
    from .multipole import AtomicMultipoles, anisotropic_energy_and_moment_potentials
    from .scc import monopole_energy
    from .calculator import ANGSTROM_TO_BOHR
    from .spin_polarization import ShellSpinPolarization

    if not isinstance(allow_large, (bool, np.bool_)):
        raise ValueError("allow_large must be a boolean")
    if len(calculator.atomic_numbers) >= 64 and not allow_large:
        raise ValueError("large Delta-SCC requires explicit allow_large=True and remote resource qualification")
    if (calculator.nn_enabled or calculator.temperature_kelvin != 0
            or calculator.alpb_parameters is not None
            or len(calculator.point_charges_electron)
            or calculator.symmetry_contract is not None):
        raise ValueError("experimental Delta-SCC requires zero-temperature standard gas-phase GFN2 without symmetry constraints")
    xyz = np.asarray(coordinates_angstrom, dtype=float)
    if xyz.shape != (len(calculator.atomic_numbers), 3) or not np.all(np.isfinite(xyz)):
        raise ValueError("Delta-SCC coordinates must be finite natoms x 3")
    if orbital_reference is not None:
        if ground_state is not None:raise ValueError('choose a stationary ground reference or a geometry-only orbital chart')
        from .orbital_geometry_reference import geometry_orbital_reference
        state=geometry_orbital_reference(calculator,xyz,orbital_reference)
    else:
        ground=ground_state
        if ground is None:
            from .response_reference import stationary_response_reference
            ground,_=stationary_response_reference(calculator,xyz)
        state=ground.orbital_state
        if not ground.converged:raise ValueError('Delta-SCC requires a converged ground reference')
    if (state is None or tuple(state.atomic_numbers)!=calculator.atomic_numbers
            or state.charge!=calculator.charge or state.multiplicity!=calculator.multiplicity
            or not np.allclose(state.coordinates_bohr,xyz*ANGSTROM_TO_BOHR,atol=1e-10,rtol=0)):
        raise ValueError('Delta-SCC requires matching molecular reference metadata')
    p0 = (state.coefficients * (state.alpha_occupations + state.beta_occupations)) @ state.coefficients.T
    kernel = calculator.density_response_kernel(state)
    model = calculator._model(state.coordinates_bohr, relocate_basis=True)
    core = model[4]
    if spin_reference is None:
        polarization = ShellSpinPolarization.from_gfn2(state, kernel.partition)
    else:
        from .spin_reference import FrozenSpinReference
        if not isinstance(spin_reference,FrozenSpinReference):
            raise TypeError('spin_reference must be a FrozenSpinReference')
        polarization = spin_reference.bind(state,kernel.partition)
    if spin_reference is not None and hasattr(spin_reference,'bind_charge_from_spin'):
        charge_correction=spin_reference.bind_charge_from_spin(state,kernel.partition,polarization)
    else:
        charge_correction=(spin_reference.bind_charge(state,kernel.partition)
                           if spin_reference is not None and hasattr(spin_reference,'bind_charge') else None)

    def functional(p,with_fock=True):
        dq, dm = kernel._observables(p - p0)
        q = kernel.reference_shell_charges + dq
        m0 = kernel.reference_multipoles
        m = AtomicMultipoles(m0.charges + dm.charges, m0.dipoles_bohr + dm.dipoles_bohr,
                             m0.quadrupoles_bohr2 + dm.quadrupoles_bohr2)
        interaction = kernel.shell_interaction
        quadratic = (interaction.evaluate(q, compute_gradient=False).energy_hartree
                     if hasattr(interaction, 'evaluate') else .5 * float(q @ interaction @ q))
        aes = anisotropic_energy_and_moment_potentials(
            m, kernel.coordinates_bohr, kernel.anisotropic_parameters,
            geometry=kernel.anisotropic_geometry)[0]
        energy = (float(np.sum(p * core)) + monopole_energy(
            q, interaction, kernel.partition, quadratic_energy_override=quadratic)
            + aes + kernel._d4(m.charges).energy_hartree)
        return (core + kernel.density_dependent_fock(p - p0) if with_fock else None), energy

    _, baseline = functional(p0)
    def callback(spin_densities):
        f, e = functional(np.sum(spin_densities, axis=0))
        spin_fock, spin_energy, _ = polarization.evaluate(spin_densities)
        if charge_correction is not None:
            dv,de=charge_correction.evaluate(np.sum(spin_densities,axis=0));f=f+dv;e=e+de
        return f[None, :, :] + spin_fock, state.total_energy_hartree + e - baseline + spin_energy
    def energy_callback(spin_densities):
        total=np.sum(spin_densities,axis=0);_,e=functional(total,with_fock=False)
        e+=polarization.energy(spin_densities) if hasattr(polarization,'energy') else polarization.evaluate(spin_densities)[1]
        if charge_correction is not None:e+=charge_correction.energy(total) if hasattr(charge_correction,'energy') else charge_correction.evaluate(total)[1]
        return float(state.total_energy_hartree+e-baseline)
    return DeltaSCFContext(calculator, state, kernel, polarization, callback, model,charge_correction,energy_only=energy_callback)


def delta_cross_overlap(reference_state, context):
    """Actual Gaussian AO cross metric; never reuse S at a displaced geometry."""
    from .basis import overlap_matrix
    from .molecular_basis import build_molecular_basis
    from .parameters import load_gfn2_parameters
    old = build_molecular_basis(reference_state.atomic_numbers, reference_state.coordinates_bohr,
                                load_gfn2_parameters())
    new = context.model[0]
    n = len(old.cartesian_orbitals)
    cross = overlap_matrix(old.cartesian_orbitals + new.cartesian_orbitals)[:n, n:]
    return old.cartesian_to_spherical.T @ cross @ new.cartesian_to_spherical


def project_delta_orbitals(reference_result, context):
    previous = reference_result.reference_state
    if (previous is None or previous.charge != context.state.charge
        or previous.model_id != context.state.model_id
        or not np.array_equal(previous.atomic_numbers,context.state.atomic_numbers)):
        raise ValueError('DeltaSCF orbital projection requires the same atoms, charge and model')
    cross = delta_cross_overlap(reference_result.reference_state, context)
    projected = []
    for c in reference_result.coefficients:
        new = np.linalg.solve(context.state.overlap, cross.T @ c)
        values, vectors = np.linalg.eigh(new.T @ context.state.overlap @ new)
        if np.min(values) < 1e-8:
            raise ValueError('DeltaSCF orbital projection lost rank')
        projected.append(new @ ((vectors*values**-.5) @ vectors.T))
    return np.array(projected)
