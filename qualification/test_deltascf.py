"""Small algebraic checks only: no molecular electronic calculations."""
import numpy as np
import pytest
from matrix_gfn2.deltascf import run_deltascf, DeltaSCFConvergenceError


def test_promoted_state_survives_aufbau_order_and_matches_energy():
    h = np.diag([-1., .2, .7])
    occ = np.array([[0, 1, 0], [1, 0, 0]])
    def functional(p):
        total = np.sum(p, axis=0)
        return np.array([h + .08*total]*2), float(np.sum(h*total) + .04*np.sum(total*total))
    result = run_deltascf(np.eye(3), np.array([np.eye(3)]*2), occ, functional)
    np.testing.assert_allclose(result.density_ao, np.diag([1, 1, 0]))
    assert result.converged
    assert result.energy_hartree == pytest.approx(-.72)
    assert result.orbital_residual < 1.e-12


def test_nonorthogonal_metric_and_reference_rotation():
    s = np.array([[1., .2, 0], [.2, 1., .1], [0, .1, 1.]])
    from scipy.linalg import eigh
    h = np.diag([-1., .2, .7])
    energies, c = eigh(h, s)
    occ = np.array([[0, 1, 0], [1, 0, 0]])
    result = run_deltascf(s, np.array([c]*2), occ, lambda p:(np.array([h]*2), float(np.sum(p*h))))
    assert np.trace(result.density_ao @ s) == pytest.approx(2)
    assert result.energy_hartree == pytest.approx(energies[0]+energies[1])


def test_failed_iteration_keeps_restart_and_original_reference(tmp_path):
    h = np.diag([-1., .2, .7])
    c = np.array([np.eye(3)]*2); occ = np.array([[0, 1, 0], [1, 0, 0]])
    with pytest.raises(DeltaSCFConvergenceError) as error:
        run_deltascf(np.eye(3), c, occ, lambda p:(np.array([h]*2), np.sum(p*h)), maximum_iterations=1)
    result = error.value.result
    path = tmp_path/'restart.npz'; result.save_checkpoint(path)
    with np.load(path) as checkpoint:
        resumed = run_deltascf(checkpoint['overlap'], checkpoint['reference_coefficients'],
                               checkpoint['occupations'], lambda p:(np.array([h]*2), np.sum(p*h)),
                               initial_coefficients=checkpoint['coefficients'])
    assert resumed.converged
    assert not result.converged


def test_invalid_occupations_and_callback():
    c = np.array([np.eye(2)]*2)
    with pytest.raises(ValueError):
        run_deltascf(np.eye(2), c, [[.5, .5], [1, 0]], lambda p:(p, 0))
    with pytest.raises(ValueError, match='callback'):
        run_deltascf(np.eye(2), c, [[0, 1], [1, 0]], lambda p:(p*np.nan, 0))


def test_spin_energy_derivatives_and_reversal_symmetry():
    from matrix_gfn2.spin_polarization import ShellSpinPolarization
    s = np.array([[1., .2], [.2, 1.]])
    model = ShellSpinPolarization(s, [0, 1], [[-.03, -.02], [-.02, -.04]])
    p = np.array([[[.7, .1], [.1, .2]], [[.2, -.1], [-.1, .5]]])
    v, energy, m = model.evaluate(p)
    np.testing.assert_allclose(v[0], -v[1])
    assert energy == pytest.approx(.5*m @ model.coupling_hartree @ m)
    assert model.evaluate(p[::-1])[1] == pytest.approx(energy)
    direction = np.array([[[.2, .3], [.3, -.1]], [[-.1, .2], [.2, .4]]])
    step = 1.e-5
    fd = (model.evaluate(p+step*direction)[1]-model.evaluate(p-step*direction)[1])/(2*step)
    assert fd == pytest.approx(float(np.sum(v*direction)), abs=1.e-11)
    v0, e0, _ = model.evaluate(np.array([p[0]]*2))
    assert e0 == 0
    assert np.max(np.abs(v0)) == 0


def test_published_constants_respect_actual_shell_order():
    from types import SimpleNamespace
    from matrix_gfn2.spin_polarization import ShellSpinPolarization, spin_constants
    # Fe is deliberately supplied as d,s,p, not the table's s,p,d ordering.
    state = SimpleNamespace(overlap=np.eye(3), atomic_numbers=np.array([26]),
                            ao_atom=np.zeros(3, int), ao_shell=('3d', '4s', '4p'))
    partition = SimpleNamespace(orbital_shell=(0, 1, 2), shell_atom=(0, 0, 0))
    model = ShellSpinPolarization.from_gfn2(state, partition)
    ss, sp, pp, sd, pd, dd = spin_constants()[25]
    np.testing.assert_allclose(model.coupling_hartree, [[dd, sd, pd], [sd, ss, sp], [pd, sp, pp]])


def test_spin_polarized_solver_splits_focks_and_relaxes_both_channels():
    from matrix_gfn2.spin_polarization import ShellSpinPolarization
    model = ShellSpinPolarization(np.eye(2), [0, 1], [[-.2, 0], [0, -.3]])
    h = np.array([[-.7, .08], [.08, -.3]])
    occ = np.array([[1, 0], [0, 1]])
    def functional(p):
        v, espin, _ = model.evaluate(p)
        return h[None]+v, float(np.sum(p*h))+espin
    result = run_deltascf(np.eye(2), np.array([np.eye(2)]*2), occ, functional,
                           maximum_iterations=500)
    assert result.converged
    assert np.max(np.abs(result.fock[0]-result.fock[1])) > .05
    assert np.max(np.abs(result.coefficients[0]-result.coefficients[1])) > .01
    assert 0 < result.spin_squared < 1.1
    assert result.energy_hartree == pytest.approx(functional(result.spin_densities_ao)[1])


def test_gfn2_driver_reuses_ground_state_and_exports_valence_determinant(monkeypatch):
    """Synthetic SCC functional: exercise the real spin driver without a QM job."""
    from types import SimpleNamespace as NS
    import matrix_gfn2.multipole as multipole
    from matrix_gfn2.deltascf import evaluate_gfn2_deltascf
    from matrix_apoc import follow_deltascf_state
    monkeypatch.setattr(multipole, 'anisotropic_energy_and_moment_potentials',
                        lambda *a, **kw: (0., None, None, None))
    m0 = multipole.AtomicMultipoles(np.array([-1., 1.]), np.zeros((2, 3)), np.zeros((2, 6)))
    partition = NS(orbital_shell=(0, 1), shell_atom=(0, 1), third_order_hartree=(0., 0.))
    class Kernel:
        reference_shell_charges = np.array([-1., 1.])
        reference_multipoles = m0
        shell_interaction = np.eye(2)*.1
        coordinates_bohr = np.zeros((2, 3))
        anisotropic_parameters = anisotropic_geometry = None
        def _observables(self, delta):
            dq = -np.diag(delta)
            return dq, multipole.AtomicMultipoles(dq, np.zeros((2, 3)), np.zeros((2, 6)))
        def density_dependent_fock(self, delta):
            return np.diag(-.1*(self.reference_shell_charges-np.diag(delta)))
        def _d4(self, q):
            return NS(energy_hartree=0.)
    kernel = Kernel(); kernel.partition = partition
    state = NS(atomic_numbers=np.array([1, 1]), charge=0, multiplicity=1,
        coordinates_bohr=np.zeros((2, 3)), coefficients=np.eye(2),
        alpha_occupations=np.array([1., 0]), beta_occupations=np.array([1., 0]),
        alpha_electrons=1., beta_electrons=1., electron_count=2, nmo=2,
        fock=np.diag([-.7, .1]), overlap=np.eye(2), total_energy_hartree=-1.5,
        ao_atom=np.array([0, 1]), ao_shell=('1s', '1s'), model_id='synthetic-SCC')
    calculator = NS(atomic_numbers=(1, 1), nn_enabled=False, temperature_kelvin=0.,
        alpb_parameters=None, point_charges_electron=[], symmetry_contract=None,
        charge=0, multiplicity=1, _valence=np.array([1, 1]),
        density_response_kernel=lambda state: kernel,
        _model=lambda *a, **kw:(None,None,None,None,np.diag([-.8,.2])))
    ground = NS(converged=True, orbital_state=state)
    result = evaluate_gfn2_deltascf(calculator, np.zeros((2, 3)), [0, 1], [1, 0], ground_state=ground)
    assert result.energy_hartree == pytest.approx(-.6-.071625)
    assert result.ground_energy_hartree == pytest.approx(-1.5)
    descriptor = result.to_apoc(2)
    assert descriptor.wavefunction.model.startswith('synthetic-SCC.spin.')
    assert follow_deltascf_state(descriptor, (descriptor,), np.eye(2))[1].overlap == 1
    triplet = evaluate_gfn2_deltascf(calculator, np.zeros((2, 3)), [1, 1], [0, 0], ground_state=ground)
    assert triplet.to_apoc().wavefunction.multiplicity == 3
    assert triplet.spin_squared == pytest.approx(2)


def test_callback_requires_two_spin_focks():
    with pytest.raises(ValueError, match='callback'):
        run_deltascf(np.eye(2), np.array([np.eye(2)]*2), [[0, 1], [1, 0]],
                    lambda p:(np.eye(2), 0.))
