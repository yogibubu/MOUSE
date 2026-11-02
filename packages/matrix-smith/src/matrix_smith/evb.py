"""Two-state EVB evaluator with analytic first and second derivatives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .surface_representation import SurfaceRepresentationRequest, validate_basin_transition


@dataclass(frozen=True)
class EVBEvaluation:
    energy: float
    gradient: np.ndarray
    state_weights: np.ndarray
    gap: float
    hessian: np.ndarray | None = None


def evaluate_two_state_evb(
    energies: tuple[float, float] | np.ndarray,
    gradients: tuple[np.ndarray, np.ndarray] | np.ndarray,
    coupling: float,
    coupling_gradient: np.ndarray | None = None,
    hessians: tuple[np.ndarray, np.ndarray] | np.ndarray | None = None,
    coupling_hessian: np.ndarray | None = None,
) -> EVBEvaluation:
    """Evaluate the lower eigenvalue of a two-state diabatic Hamiltonian.

    Supplying diabatic Hessians requests the analytic adiabatic Hessian.  The
    coupling Hessian defaults to zero only when the coupling gradient is also
    absent; otherwise it must be supplied explicitly so a kinetic saddle is
    never certified from an incomplete second derivative.
    """

    diagonal = np.asarray(energies, dtype=float).reshape(-1)
    if diagonal.shape != (2,) or not np.all(np.isfinite(diagonal)):
        raise ValueError("EVB requires two finite diabatic energies")
    grad = np.asarray(gradients, dtype=float)
    if grad.ndim != 2 or grad.shape[0] != 2 or not np.all(np.isfinite(grad)):
        raise ValueError("EVB gradients must have shape (2, n) and be finite")
    coupling = float(coupling)
    if not np.isfinite(coupling):
        raise ValueError("EVB coupling must be finite")
    matrix = np.asarray(((diagonal[0], coupling), (coupling, diagonal[1])), dtype=float)
    values, vectors = np.linalg.eigh(matrix)
    coefficients = vectors[:, 0]
    weights = coefficients * coefficients
    gap = float(values[1] - values[0])
    dimension = grad.shape[1]
    coupling_first = np.zeros(dimension, dtype=float)
    if coupling_gradient is not None:
        derivative = np.asarray(coupling_gradient, dtype=float).reshape(-1)
        if derivative.shape != (dimension,) or not np.all(np.isfinite(derivative)):
            raise ValueError("EVB coupling gradient has invalid shape or values")
        coupling_first = derivative

    mean_first = 0.5 * (grad[0] + grad[1])
    delta = 0.5 * (diagonal[0] - diagonal[1])
    delta_first = 0.5 * (grad[0] - grad[1])
    radius = float(np.hypot(delta, coupling))
    if radius <= np.finfo(float).eps:
        raise ValueError("EVB adiabatic derivatives are undefined at an exact uncoupled crossing")
    numerator_first = delta * delta_first + coupling * coupling_first
    gradient = mean_first - numerator_first / radius

    hessian = None
    if hessians is not None:
        second = np.asarray(hessians, dtype=float)
        if second.shape != (2, dimension, dimension) or not np.all(np.isfinite(second)):
            raise ValueError("EVB Hessians must have shape (2, n, n) and be finite")
        if not np.allclose(second, np.swapaxes(second, 1, 2), rtol=1.0e-11, atol=1.0e-13):
            raise ValueError("EVB diabatic Hessians must be symmetric")
        if coupling_hessian is None:
            if coupling_gradient is not None:
                raise ValueError(
                    "EVB coupling Hessian is required when a coupling gradient is supplied"
                )
            coupling_second = np.zeros((dimension, dimension), dtype=float)
        else:
            coupling_second = np.asarray(coupling_hessian, dtype=float)
            if (
                coupling_second.shape != (dimension, dimension)
                or not np.all(np.isfinite(coupling_second))
                or not np.allclose(
                    coupling_second,
                    coupling_second.T,
                    rtol=1.0e-11,
                    atol=1.0e-13,
                )
            ):
                raise ValueError("EVB coupling Hessian must be finite and symmetric")
        mean_second = 0.5 * (second[0] + second[1])
        delta_second = 0.5 * (second[0] - second[1])
        radius_second_numerator = (
            np.outer(delta_first, delta_first)
            + delta * delta_second
            + np.outer(coupling_first, coupling_first)
            + coupling * coupling_second
        )
        radius_hessian = (
            radius_second_numerator / radius
            - np.outer(numerator_first, numerator_first) / radius**3
        )
        hessian = mean_second - radius_hessian
        hessian = 0.5 * (hessian + hessian.T)

    return EVBEvaluation(float(values[0]), gradient, weights, gap, hessian)


def evaluate_two_state_evb_request(
    request: SurfaceRepresentationRequest,
    energies: tuple[float, float] | np.ndarray,
    gradients: tuple[np.ndarray, np.ndarray] | np.ndarray,
    coupling: float,
    coupling_gradient: np.ndarray | None = None,
    hessians: tuple[np.ndarray, np.ndarray] | np.ndarray | None = None,
    coupling_hessian: np.ndarray | None = None,
) -> EVBEvaluation:
    """Evaluate EVB only when the explicit representation/gate contract is present."""

    if request.provider != "EVB":
        raise ValueError("EVB evaluator requires provider='EVB'")
    validate_basin_transition(request)
    return evaluate_two_state_evb(
        energies,
        gradients,
        coupling,
        coupling_gradient,
        hessians,
        coupling_hessian,
    )
