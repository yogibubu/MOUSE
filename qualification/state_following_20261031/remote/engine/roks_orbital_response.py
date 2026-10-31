"""Local ROKS orbital Newton equations with the existing analytic Fock kernel."""
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
from scipy.linalg import expm


def orbital_residual_jacobian(c, occupations, focks, kernel, polarization, left, right):
    """Derivative of the moving-frame orbital residual, without differences."""
    n = len(c)
    dc = np.zeros((len(left), n, n))
    for b, (i, j) in enumerate(zip(left, right)):
        dc[b, :, j] = c[:, i]
        dc[b, :, i] = -c[:, j]
    derivative = np.zeros((len(left), len(left)))
    for spin, (o, f) in enumerate(zip(occupations, focks)):
        dp = (dc*o) @ c.T
        dp += dp.transpose(0, 2, 1)
        # The common density derivative includes BOTH spin channels.
        other = (dc*occupations[1-spin]) @ c.T
        total = dp+other+other.transpose(0, 2, 1)
        df = []
        for b in range(len(left)):
            spin_dp = np.zeros((2, n, n))
            spin_dp[spin] = dp[b]
            spin_dp[1-spin] = other[b]+other[b].T
            spin_fock, _, _ = polarization.evaluate(spin_dp)
            df.append(kernel(total[b])+spin_fock[spin])
        mo = dc.transpose(0, 2, 1) @ f @ c + c.T @ f @ dc + c.T @ np.asarray(df) @ c
        derivative += (2*mo[:, left, right]*(o[right]-o[left])).T
    return derivative


def run_roks_orbital_newton(context, coefficients, occupations, *,
                            orbital_tolerance=1e-9, maximum_evaluations=100,
                            maximum_step=.5, certify_index=False, trial_guard=None):
    from .roks import ROKSSingletResult, ROKSConvergenceError
    c = np.array(coefficients, copy=True)
    occ = np.asarray(occupations)
    triplet = np.array([np.maximum(*occ), np.minimum(*occ)])
    left, right = np.triu_indices(len(c), 1)
    labels = np.vstack((occ, triplet))
    active = np.any(labels[:, left] != labels[:, right], axis=0)
    left, right = left[active], right[active]
    count = 0

    def evaluate(coeff):
        nonlocal count
        count += 1
        pm = np.array([(coeff*o) @ coeff.T for o in occ])
        pt = np.array([(coeff*o) @ coeff.T for o in triplet])
        fm, em = context.fock_and_energy(pm)
        ft, et = context.fock_and_energy(pt)
        q = sum(2*(coeff.T @ f @ coeff)*o[None, :] for f, o in zip(fm, occ))
        q -= sum((coeff.T @ f @ coeff)*o[None, :] for f, o in zip(ft, triplet))
        return 2*(q[left, right]-q[right, left]), fm, ft, 2*em-et

    residual, fm, ft, energy = evaluate(c)
    while count < maximum_evaluations and np.max(abs(residual), initial=0.) > orbital_tolerance:
        jacobian = np.zeros((len(left), len(left)))
        for weight, o, f in ((2., occ, fm), (-1., triplet, ft)):
            # A fixed-density derivative container; no ground-state SCF call.
            state = SimpleNamespace(**vars(context.state))
            state.coefficients = c
            state.alpha_occupations, state.beta_occupations = o
            kernel = context.density_response_kernel(state)
            jacobian += weight*orbital_residual_jacobian(c, o, f, kernel,
                context.polarization, left, right)
        step = np.linalg.lstsq(jacobian, -residual, rcond=1e-10)[0]
        norm = np.linalg.norm(step)
        if norm > maximum_step:
            step *= maximum_step/norm
        accepted = False
        for fraction in (1., .5, .25, .125, .0625, .03125):
            if count >= maximum_evaluations:
                break
            k = np.zeros_like(c)
            k[left, right] = fraction*step
            k[right, left] = -fraction*step
            trial = c @ expm(k)
            if trial_guard is not None and not trial_guard(c, trial):
                continue
            rr, ffm, fft, ee = evaluate(trial)
            if np.linalg.norm(rr) < np.linalg.norm(residual):
                c, residual, fm, ft, energy = trial, rr, ffm, fft, ee
                accepted = True
                break
        if not accepted:
            break
    error = float(np.max(abs(residual), initial=0.))
    result = ROKSSingletResult(c, occ, triplet, context.state.overlap, fm, ft,
        float(energy), error, count, error <= orbital_tolerance)
    if not result.converged:
        raise ROKSConvergenceError(result)
    if certify_index:
        jacobian = np.zeros((len(left), len(left)))
        for weight, o, f in ((2., occ, fm), (-1., triplet, ft)):
            state = SimpleNamespace(**vars(context.state))
            state.coefficients = c
            state.alpha_occupations, state.beta_occupations = o
            kernel = context.density_response_kernel(state)
            jacobian += weight*orbital_residual_jacobian(c, o, f, kernel,
                context.polarization, left, right)
        curvature = np.linalg.eigvalsh(.5*(jacobian+jacobian.T))
        if np.any(abs(curvature) <= 1e-7):
            raise ValueError('stationary ROKS branch has an unresolved orbital index')
        index = int(np.count_nonzero(curvature < -1e-7))
        result = replace(result,orbital_saddle_index=index,target_orbital_saddle_index=index)
    return result


def run_roks_gmf(context, coefficients, occupations, *, saddle_index,
                 orbital_tolerance=1e-9, maximum_iterations=100,
                 maximum_step=.2, freeze_and_release=False,
                 freeze_iterations=12, checkpoint=None, trial_guard=None):
    """Find a specified electronic saddle using analytic orbital curvature.

    freeze_and_release relaxes the other orbitals with the two open orbitals
    frozen before releasing all active rotations. The requested saddle index
    applies to the full shared-orbital ROKS manifold, not the nuclear Hessian.
    With saddle_index=None, infer the index from the analytic curvature AFTER
    a converged frozen-orbital relaxation (FR-DO); never infer it from energies
    or from the unrelaxed promoted determinant.
    """
    from .orbital_saddle import generalized_mode_following
    from .roks import ROKSSingletResult, ROKSConvergenceError
    if saddle_index is None and not freeze_and_release:
        raise ValueError('automatic ROKS saddle index requires freeze-and-release')
    occ = np.asarray(occupations,dtype=float)
    c0 = np.asarray(coefficients,dtype=float)
    triplet = np.array([np.maximum(*occ), np.minimum(*occ)])
    left,right = np.triu_indices(len(c0),1)
    labels = np.vstack((occ,triplet))
    active = np.any(labels[:,left] != labels[:,right],axis=0)
    left,right = left[active],right[active]
    evaluations = 0
    cache = {}

    def objective(c,curvature=True):
        nonlocal evaluations
        if 'c' not in cache or not np.array_equal(c,cache['c']):
            pm = np.array([(c*o) @ c.T for o in occ])
            pt = np.array([(c*o) @ c.T for o in triplet])
            fm,em = context.fock_and_energy(pm)
            ft,et = context.fock_and_energy(pt)
            evaluations += 2
            q = sum(2*(c.T @ f @ c)*o[None,:] for f,o in zip(fm,occ))
            q -= sum((c.T @ f @ c)*o[None,:] for f,o in zip(ft,triplet))
            g = 2*(q[left,right]-q[right,left])
            cache.clear()
            cache.update(c=c.copy(),fm=fm,ft=ft,energy=float(2*em-et),g=g)
        if curvature and 'h' not in cache:
            # Mixed and triplet components have the SAME total AO density.
            # Their common charge-response kernel is therefore identical.
            state = SimpleNamespace(**vars(context.state))
            state.coefficients = c
            state.alpha_occupations,state.beta_occupations = occ
            kernel = context.density_response_kernel(state)
            j = 2*orbital_residual_jacobian(c,occ,cache['fm'],kernel,
                context.polarization,left,right)
            j -= orbital_residual_jacobian(c,triplet,cache['ft'],kernel,
                context.polarization,left,right)
            cache['h'] = .5*(j+j.T)
        return cache['energy'],cache['g'],cache.get('h')

    def retract(c,step,indices=None):
        k = np.zeros_like(c)
        il,jl = (left,right) if indices is None else (left[indices],right[indices])
        k[il,jl] = step
        k[jl,il] = -step
        return c @ expm(k)

    if freeze_and_release:
        opens = np.flatnonzero(occ[0] != occ[1])
        allowed = np.flatnonzero(~np.isin(left,opens) & ~np.isin(right,opens))
        if len(allowed):
            def restricted(c,curvature=True):
                e,g,h = objective(c,curvature=curvature)
                return e,g[allowed],None if h is None else h[np.ix_(allowed,allowed)]
            frozen = generalized_mode_following(restricted,
                lambda c,p:retract(c,p,allowed),c0,saddle_index=0,
                gradient_tolerance=max(orbital_tolerance,1e-6),
                maximum_iterations=freeze_iterations,maximum_step=maximum_step,
                trial_guard=trial_guard)
            if saddle_index is None and not frozen.converged:
                raise ValueError('cannot infer saddle index before frozen relaxation converges')
            c0 = frozen.point
    if saddle_index is None:
        _, _, h = objective(c0,curvature=True)
        values = np.linalg.eigvalsh(h)
        if np.any(abs(values)<=1e-7):
            raise ValueError('frozen relaxed orbital curvature cannot resolve saddle index')
        saddle_index = int(np.count_nonzero(values < -1e-7))
    solved = generalized_mode_following(objective,retract,c0,
        saddle_index=saddle_index,gradient_tolerance=orbital_tolerance,
        maximum_iterations=maximum_iterations,maximum_step=maximum_step,
        checkpoint=checkpoint,trial_guard=trial_guard)
    energy,g,_ = objective(solved.point,curvature=False)
    result = ROKSSingletResult(solved.point,occ,triplet,context.state.overlap,
        cache['fm'],cache['ft'],energy,float(np.max(abs(g),initial=0.)),
        evaluations,solved.converged,orbital_saddle_index=solved.observed_index,
        target_orbital_saddle_index=solved.target_index)
    if not result.converged:
        raise ROKSConvergenceError(result)
    return result
