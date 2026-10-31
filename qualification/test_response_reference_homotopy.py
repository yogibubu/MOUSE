"""Molecular regression: Newton recertification must not replay hot annealing."""
import numpy as np
import pytest
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.response_reference import stationary_response_reference
from matrix_gfn2.calculator import _annealed_scf_temperature

XYZ=np.array([[0.03656918131967183,-6.585047596976614e-16,1.2216860839300792],
 [-0.10690863395942318,-2.4260250183666e-16,-0.0004046195304328277],
 [0.5703392526236906,2.4864350118936553e-16,-0.7212816544157092]])


def test_temperature_control_preserves_target_and_default_homotopy():
    assert _annealed_scf_temperature(0.,1,.0099)==2000.
    assert _annealed_scf_temperature(0.,13,.0099)==0.
    assert _annealed_scf_temperature(0.,1,.0099,enabled=False)==0.
    assert _annealed_scf_temperature(300.,1,.2,enabled=False)==300.
    with pytest.raises(ValueError):_annealed_scf_temperature(-1.,1,.0099,enabled=False)


@pytest.mark.parametrize('energy_tolerance',[1e-6,1e-10])
def test_small_gap_hno_reference_closes_without_relaxing_gates(monkeypatch,energy_tolerance):
    import matrix_gfn2.calculator as module
    original=module._annealed_scf_temperature
    def old_reannealing(target,iteration,gap,*,enabled=True):
        return original(target,iteration,gap)  # Reproduce the original defect.
    monkeypatch.setattr(module,'_annealed_scf_temperature',old_reannealing)
    calc=GFN2Calculator(['O','N','H'],temperature_kelvin=0.,scf_energy_tolerance_hartree=energy_tolerance,maximum_scf_iterations=500)
    # Actual retained molecular atom order is checked below in the driver.
    with pytest.raises(RuntimeError,match='Newton correction did not close'):
        stationary_response_reference(calc,XYZ)
    monkeypatch.setattr(module,'_annealed_scf_temperature',original)
    settings=(calc.temperature_kelvin,calc.scf_energy_tolerance_hartree,calc.scf_wavefunction_tolerance_electron,calc.scf_mixer_tolerance)
    result,diagnostics=stationary_response_reference(calc,XYZ)
    assert result.converged and diagnostics['newton_corrections']>0
    assert diagnostics['fock_closure_residuals_hartree'][-1]<1e-10
    assert diagnostics['canonical_eigen_residuals_hartree'][-1]<1e-12
    assert settings==(calc.temperature_kelvin,calc.scf_energy_tolerance_hartree,calc.scf_wavefunction_tolerance_electron,calc.scf_mixer_tolerance)
    assert diagnostics['correction_temperature_policy']=='requested_temperature_without_reannealing'
    state=result.orbital_state
    kernel=calc.density_response_kernel(state)
    fock=calc._model(state.coordinates_bohr,relocate_basis=True)[4]+kernel.density_dependent_fock(np.zeros_like(state.fock))
    assert np.max(abs(fock@state.coefficients-state.overlap@state.coefficients*state.orbital_energies_hartree))<1e-12


def test_result_cache_distinguishes_temperature_homotopy(monkeypatch):
    calc=GFN2Calculator(['O','N','H'],temperature_kelvin=0.)
    result=calc.evaluate(XYZ,retain_orbitals=True,compute_bond_orders=False)
    calls=[]
    def single(*a,**kw):calls.append(kw.get('temperature_homotopy',True));return result
    monkeypatch.setattr(calc,'_single_point',single)
    for enabled in [False,False,True,True,False]:
        calc.evaluate(XYZ,retain_orbitals=True,compute_bond_orders=False,temperature_homotopy=enabled)
    assert calls==[False,True,False]
    with pytest.raises(ValueError,match='boolean'):calc.evaluate(XYZ,temperature_homotopy='false')


def test_newton_restart_tolerates_crossing_energy_guesses(monkeypatch):
    from dataclasses import replace
    import matrix_gfn2.ground_state_response as response
    original=response.solve_ground_state_directional_response
    def crossing_guesses(state,*a,**kw):
        result=original(state,*a,**kw)
        shift=result.orbital_energy_derivative.copy()
        occupied=int(np.sum(state.alpha_occupations))
        shift[occupied-1]+=1.  # Density Newton correction remains unchanged.
        return replace(result,orbital_energy_derivative=shift)
    monkeypatch.setattr(response,'solve_ground_state_directional_response',crossing_guesses)
    calc=GFN2Calculator(['O','N','H'],temperature_kelvin=0.)
    ground,diagnostics=stationary_response_reference(calc,XYZ)
    assert ground.converged and diagnostics['canonical_eigen_residuals_hartree'][-1]<1e-12
    assert np.all(np.diff(ground.orbital_state.orbital_energies_hartree)>=0.)


def test_unconverged_warm_scf_reports_convergence_not_false_metric_failure():
    calc=GFN2Calculator(['O','N','H'],temperature_kelvin=0.)
    calc.evaluate(XYZ,retain_orbitals=True,compute_bond_orders=False)
    calc.maximum_scf_iterations=1
    displaced=XYZ.copy();displaced[0,0]+=.02
    with pytest.raises(RuntimeError,match='ISCF did not converge'):
        calc.evaluate(displaced,retain_orbitals=True,compute_bond_orders=False)


def test_composite_optimizer_finish_does_not_launch_ground_descriptor_scf(tmp_path):
    from types import SimpleNamespace
    from matrix_link.optimizer import _write_final_matrix_gfn2_scaling_observables
    def forbidden(*a,**kw):raise AssertionError('unrelated S0 descriptor solve after ROKS optimization')
    service=SimpleNamespace(backend=SimpleNamespace(name='matrix_gfn2',electronic_state=1,native_excited_method='roks_50_50'),
        _matrix_gfn2_session=SimpleNamespace(write_scaling_observables=forbidden))
    assert _write_final_matrix_gfn2_scaling_observables(service,XYZ,tmp_path) is None
