"""Dense, index-targeted generalized mode following in orbital space.

The modified gradient reflects its components along the lowest m Hessian
eigenvectors (Schmerwitz, Levi and Jonsson, JCTC 2023, 19, 3634).
This is an orbital saddle search, not a minimization of the squared residual.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class OrbitalSaddleResult:
    point: object
    energy: float
    gradient: np.ndarray
    eigenvalues: np.ndarray
    target_index: int
    observed_index: int
    converged: bool
    iterations: int
    history: tuple


def generalized_mode_following(evaluate, retract, initial, *, saddle_index,
                               gradient_tolerance=1e-9, maximum_iterations=100,
                               maximum_step=.2, curvature_floor=1e-3,
                               index_tolerance=1e-7, checkpoint=None,
                               trial_guard=None):
    """Follow a fixed Hessian index using analytic gradients and Hessians.

    evaluate(point, curvature=True) returns (energy, gradient, Hessian).
    With curvature=False the last return value may be None. A bounded step
    can increase the gradient norm; forbidding that would recreate the
    spurious minima of squared-gradient minimization. Only excessive growth
    is rejected. Both stationarity AND the requested index certify success.
    An optional fixed-geometry trial guard rejects state-changing steps before
    evaluating them; it never adds a penalty or changes these certificates.
    """
    if trial_guard is not None and not callable(trial_guard):
        raise ValueError('orbital trial guard must be callable')
    if (isinstance(saddle_index, bool) or not isinstance(saddle_index, (int, np.integer))
            or saddle_index < 0 or maximum_iterations < 1
            or not all(np.isfinite(x) and x > 0 for x in
                       (gradient_tolerance, maximum_step, curvature_floor, index_tolerance))):
        raise ValueError('invalid orbital saddle-search controls')
    point = initial
    trust = float(maximum_step)
    history = []
    for iteration in range(maximum_iterations+1):
        energy, g, h = evaluate(point, curvature=True)
        g, h = np.asarray(g), np.asarray(h)
        if (g.ndim != 1 or h.shape != (len(g), len(g)) or saddle_index > len(g)
                or not np.isfinite(energy) or not np.all(np.isfinite(g)) or not np.all(np.isfinite(h))
                or not np.allclose(h, h.T, atol=1e-8, rtol=0)):
            raise ValueError('invalid analytic orbital saddle objective')
        values, vectors = np.linalg.eigh(.5*(h+h.T))
        observed = int(np.count_nonzero(values < -index_tolerance))
        error = float(np.max(abs(g), initial=0.))
        flat = bool(np.any(abs(values) <= index_tolerance))
        converged = error <= gradient_tolerance and observed == saddle_index and not flat
        row = dict(iteration=iteration, energy=float(energy), residual=error,
                   observed_index=observed, target_index=int(saddle_index),
                   minimum_curvature=float(values[0]) if len(values) else None,
                   trust_radius=trust)
        history.append(row)
        result = OrbitalSaddleResult(point, float(energy), g, values, int(saddle_index),
                                    observed, converged, iteration, tuple(history))
        if checkpoint is not None:
            checkpoint(result)
        if converged or iteration == maximum_iterations or not len(g):
            return result
        reflected = vectors.T @ g
        reflected[:saddle_index] *= -1
        step = -vectors @ (reflected/np.maximum(abs(values), curvature_floor))
        if error <= gradient_tolerance and observed != saddle_index:
            # An exactly stationary point with the wrong index cannot be
            # accepted; leave along a mode whose required sign is incorrect.
            mode = min(observed, saddle_index)
            step = .05*vectors[:, mode]
        norm = np.linalg.norm(step)
        if norm > trust:
            step *= trust/norm
        accepted = False
        for fraction in (1., .5, .25, .125, .0625):
            trial = retract(point, fraction*step)
            if trial_guard is not None and not trial_guard(point, trial):
                continue
            trial_energy, trial_g, _ = evaluate(trial, curvature=False)
            trial_norm = np.linalg.norm(trial_g)
            growth_bound = max(4*np.linalg.norm(g), 10*gradient_tolerance)
            if error <= gradient_tolerance and observed != saddle_index:
                growth_bound = max(growth_bound, .1)
            if (np.isfinite(trial_energy) and np.isfinite(trial_norm)
                    and trial_norm <= growth_bound):
                point = trial
                accepted = True
                # Leaving a basin with the wrong curvature index normally
                # INCREASES |g| before reaching the target saddle. Shrinking
                # every accepted uphill step can make the total remaining
                # travel finite and strand the search inside that basin.
                factor=1.1 if observed!=saddle_index else (1.2 if trial_norm<np.linalg.norm(g) else .8)
                trust = min(maximum_step, trust*factor)
                break
        if not accepted:
            trust *= .5
        if trust < 1e-8:
            return result
    return result
