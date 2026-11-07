import numpy as np
import pytest

from matrix_smith.evb import evaluate_two_state_evb


def _diabatic(point):
    x, y = np.asarray(point, dtype=float)
    h0 = np.array([[1.2, 0.15], [0.15, 0.8]])
    h1 = np.array([[0.7, -0.10], [-0.10, 1.1]])
    g0 = np.array([0.30, -0.20]) + h0 @ np.array([x, y])
    g1 = np.array([-0.25, 0.35]) + h1 @ np.array([x, y])
    e0 = 0.2 + np.dot([0.30, -0.20], [x, y]) + 0.5 * np.dot([x, y], h0 @ [x, y])
    e1 = 0.5 + np.dot([-0.25, 0.35], [x, y]) + 0.5 * np.dot([x, y], h1 @ [x, y])
    hc = np.array([[0.08, 0.02], [0.02, -0.04]])
    gc = np.array([0.05, -0.03]) + hc @ np.array([x, y])
    coupling = 0.12 + np.dot([0.05, -0.03], [x, y]) + 0.5 * np.dot([x, y], hc @ [x, y])
    return (e0, e1), np.stack((g0, g1)), np.stack((h0, h1)), coupling, gc, hc


def _evaluate(point):
    energies, gradients, hessians, coupling, coupling_gradient, coupling_hessian = _diabatic(point)
    return evaluate_two_state_evb(
        energies,
        gradients,
        coupling,
        coupling_gradient,
        hessians,
        coupling_hessian,
    )


def test_analytic_evb_hessian_matches_gradient_difference():
    point = np.array([0.13, -0.08])
    result = _evaluate(point)
    step = 2.0e-5
    numeric = np.empty((2, 2))
    for column in range(2):
        displacement = np.zeros(2)
        displacement[column] = step
        numeric[:, column] = (
            _evaluate(point + displacement).gradient
            - _evaluate(point - displacement).gradient
        ) / (2.0 * step)
    np.testing.assert_allclose(result.hessian, numeric, rtol=2.0e-8, atol=2.0e-10)
    np.testing.assert_allclose(result.hessian, result.hessian.T, atol=1.0e-14)


def test_gradient_dependent_coupling_requires_coupling_hessian_for_second_order():
    with pytest.raises(ValueError, match="coupling Hessian"):
        evaluate_two_state_evb(
            (0.0, 0.2),
            np.zeros((2, 1)),
            0.1,
            np.ones(1),
            np.zeros((2, 1, 1)),
        )
