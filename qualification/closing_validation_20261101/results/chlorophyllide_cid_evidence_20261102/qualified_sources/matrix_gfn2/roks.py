"""Two-open-orbital spin-adapted ROKS on the spin-polarized GFN2 functional.

The singlet CSF is pure S=0 by construction. Its approximate functional energy
is the multiplet sum 2 E_M - E_T evaluated with ONE shared spatial orbital set;
this is not an exact projection energy of independently relaxed UHF orbitals.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares

from .deltascf import prepare_gfn2_deltascf, project_delta_orbitals, delta_cross_overlap, run_deltascf


@dataclass(frozen=True)
class ROKSSingletResult:
    common_coefficients: np.ndarray
    occupations: np.ndarray
    triplet_occupations: np.ndarray
    overlap: np.ndarray
    fock_mixed: np.ndarray
    fock_triplet: np.ndarray
    energy_hartree: float
    orbital_residual: float
    evaluations: int
    converged: bool
    ground_energy_hartree: float | None = None
    reference_state: object | None = None
    core_electrons: np.ndarray | None = None
    gradient_hartree_per_bohr: np.ndarray | None = None
    orbital_saddle_index: int | None = None
    target_orbital_saddle_index: int | None = None
    spin_reference_fingerprint: str | None = None
    spin_frame_clusters: np.ndarray | None = None

    def __post_init__(self):
        indices=(self.orbital_saddle_index,self.target_orbital_saddle_index)
        if ((indices[0] is None) != (indices[1] is None)
                or any(i is not None and (isinstance(i,(bool,np.bool_))
                       or not isinstance(i,(int,np.integer)) or i < 0) for i in indices)
                or (self.converged and indices[0] != indices[1])):
            raise ValueError('inconsistent ROKS orbital saddle-index certificate')
        c=np.asarray(self.common_coefficients,dtype=float)
        s=np.asarray(self.overlap,dtype=float)
        o=np.asarray(self.occupations,dtype=float)
        t=np.asarray(self.triplet_occupations,dtype=float)
        if (s.ndim!=2 or s.shape[0]!=s.shape[1] or c.shape!=s.shape
            or o.shape!=(2,len(s)) or t.shape!=o.shape
            or not all(np.all(np.isfinite(a)) for a in (s,c,o,t))
            or not np.all((o==0)|(o==1))
            or np.count_nonzero(o[0]-o[1]==1)!=1
            or np.count_nonzero(o[1]-o[0]==1)!=1
            or not np.array_equal(t,np.array([np.maximum(*o),np.minimum(*o)]))
            or not np.allclose(c.T @ s @ c,np.eye(len(s)),atol=1e-8,rtol=0)
            or not np.isfinite(self.energy_hartree)
            or not np.isfinite(self.orbital_residual) or self.orbital_residual<0):
            raise ValueError('invalid common-orbital spin-pure ROKS result')
        for name in ('common_coefficients','occupations','triplet_occupations',
                     'overlap','fock_mixed','fock_triplet','core_electrons','gradient_hartree_per_bohr','spin_frame_clusters'):
            value=getattr(self,name)
            if value is None:
                continue
            value=np.array(value,copy=True)
            if not np.all(np.isfinite(value)):
                raise ValueError('ROKS result arrays must be finite')
            if name.startswith('fock_') and (value.shape!=(2,len(s),len(s))
                or not np.allclose(value,value.transpose(0,2,1),atol=1e-10,rtol=0)):
                raise ValueError('invalid spin-resolved ROKS Fock matrices')
            value.setflags(write=False)
            object.__setattr__(self,name,value)

    @property
    def coefficients(self):
        return np.array([self.common_coefficients]*2)

    @property
    def spin_squared(self):
        # The CSF consists of two spin-flipped, orthogonal, common-orbital determinants.
        return 0.

    @property
    def excitation_energy_hartree(self):
        return None if self.ground_energy_hartree is None else self.energy_hartree-self.ground_energy_hartree

    @property
    def density_ao(self):
        return np.sum(self.densities(),axis=0)

    @property
    def spin_densities_ao(self):
        # The S=0 CSF has equal spin-resolved one-particle densities.
        return np.array([.5*self.density_ao]*2)

    def densities(self, triplet=False):
        """Auxiliary mixed/triplet determinant densities used by the functional."""
        occ = self.triplet_occupations if triplet else self.occupations
        c = self.common_coefficients
        return np.array([(c*o) @ c.T for o in occ])

    def to_apoc(self, state_id=1):
        from matrix_apoc import DeltaSCFState, DeterminantOrbitals, SpinAdaptedSinglet
        from .spin_polarization import SPIN_PARAMETER_SHA256
        state = self.reference_state
        if state is None:
            raise ValueError('ROKS APOC export requires molecular metadata')
        mixed = DeltaSCFState(state_id, DeterminantOrbitals(
            state.atomic_numbers, state.coordinates_bohr, self.overlap,
            tuple(self.coefficients), tuple(self.occupations), self.core_electrons,
            charge=state.charge, multiplicity=1, total_energy_hartree=self.energy_hartree,
            model=state.model_id+'.spin.'+(self.spin_reference_fingerprint or SPIN_PARAMETER_SHA256)))
        return SpinAdaptedSinglet(state_id, mixed, self.energy_hartree)


class ROKSConvergenceError(RuntimeError):
    def __init__(self, result):
        self.result = result
        super().__init__(f'ROKS orbital stationarity failed: residual={result.orbital_residual:.3e}; '
                         f'orbital index={result.orbital_saddle_index}, target={result.target_orbital_saddle_index}')


def run_roks_singlet(overlap, reference_coefficients, occupations, fock_and_energy, *,
                      orbital_tolerance=1e-9, maximum_evaluations=400, strict=True,
                      rechart_interval=0):
    """Converge common-orbital stationarity using analytic orbital residuals.

    Least-squares solves the saddle-point equations, rather than minimizing
    the excited-state energy. Its Jacobian is numerically differentiated in
    orbital-rotation space; this does not evaluate displaced nuclear points.
    A positive rechart_interval periodically resets the exponential chart at
    the current orbitals, avoiding poorly conditioned large rotation vectors.
    """
    s, c0, occ = (np.asarray(a, dtype=float) for a in (overlap, reference_coefficients, occupations))
    n = len(s)
    if (s.shape != (n, n) or c0.shape != (n, n) or occ.shape != (2, n)
            or not all(np.all(np.isfinite(a)) for a in (s,c0,occ))
            or not np.all((occ == 0) | (occ == 1))
            or np.count_nonzero(occ[0]-occ[1] == 1) != 1
            or np.count_nonzero(occ[1]-occ[0] == 1) != 1
            or not np.allclose(c0.T @ s @ c0, np.eye(n), atol=1e-7, rtol=0)
            or not np.isfinite(orbital_tolerance) or orbital_tolerance <= 0
            or maximum_evaluations < 1):
        raise ValueError('ROKS singlet requires one alpha-only and one beta-only orbital in a common orthonormal basis')
    if isinstance(rechart_interval, bool) or int(rechart_interval) != rechart_interval or rechart_interval < 0:
        raise ValueError('rechart interval must be a nonnegative integer')
    triplet = np.array([np.maximum(*occ), np.minimum(*occ)])
    left, right = np.triu_indices(n, 1)
    labels = np.vstack((occ, triplet))
    active = np.any(labels[:, left] != labels[:, right], axis=0)
    left, right = left[active], right[active]
    cache = {}
    def evaluate(x):
        if 'x' in cache and np.array_equal(x, cache['x']):
            return cache['value']
        rotation = np.zeros((n,n));rotation[left,right] = x;rotation[right,left] = -x
        c = c0 @ expm(rotation)
        pm = np.array([(c*o) @ c.T for o in occ])
        pt = np.array([(c*o) @ c.T for o in triplet])
        fm, em = fock_and_energy(pm); ft, et = fock_and_energy(pt)
        fm, ft = np.asarray(fm), np.asarray(ft)
        if (fm.shape != (2,n,n) or ft.shape != (2,n,n)
                or not all(np.all(np.isfinite(a)) for a in (fm,ft))
                or not np.isfinite(em) or not np.isfinite(et)):
            raise ValueError('invalid spin-resolved ROKS callback')
        q = sum(2*(c.T @ f @ c)*o[None,:] for f,o in zip(fm,occ,strict=True))
        q -= sum((c.T @ f @ c)*o[None,:] for f,o in zip(ft,triplet,strict=True))
        residual = 2*(q[left,right]-q[right,left])
        value = (residual, c, fm, ft, 2*em-et)
        cache.update(x=x.copy(),value=value)
        return value
    evaluations = 0
    while evaluations < maximum_evaluations:
        budget = maximum_evaluations-evaluations
        if rechart_interval:
            budget = min(budget, int(rechart_interval))
        solved = least_squares(lambda x:evaluate(x)[0], np.zeros(len(left)),
            max_nfev=budget, ftol=1e-13, xtol=1e-13, gtol=1e-13)
        evaluations += solved.nfev
        residual,c,fm,ft,energy = evaluate(solved.x)
        if np.max(np.abs(residual),initial=0.) <= orbital_tolerance or not rechart_interval:
            break
        c0 = c
        cache.clear()
    error = float(np.max(np.abs(residual),initial=0.))
    result = ROKSSingletResult(c,occ.copy(),triplet,s.copy(),fm,ft,float(energy),error,
                               evaluations,error <= orbital_tolerance)
    if strict and not result.converged:
        raise ROKSConvergenceError(result)
    return result


def evaluate_gfn2_roks_singlet(calculator, coordinates_angstrom, alpha_occupations,
                                beta_occupations, *, ground_state=None, gradient=False,
                                reference_result=None, minimum_state_overlap=.70,
                                orbital_solver='least_squares',
                                spin_reference=None,reference_policy='stationary_ground',
                                **solver_options):
    """Reoptimize a two-determinant S=0 state, optionally seeded by a BS result."""
    from .spin_reference import validate_spin_reference
    if reference_result is not None:
        validate_spin_reference(reference_result,spin_reference)
    if reference_policy not in {'stationary_ground','geometry_only'}:raise ValueError('unknown ROKS reference policy')
    if reference_policy=='geometry_only' and (reference_result is None or ground_state is not None):
        raise ValueError('geometry-only ROKS requires a retained molecular state and no ground-state solve')
    context = prepare_gfn2_deltascf(calculator,coordinates_angstrom,ground_state=ground_state,
        spin_reference=spin_reference,orbital_reference=reference_result.reference_state if reference_policy=='geometry_only' else None)
    state, callback = context.state, context.fock_and_energy
    occ = np.asarray([alpha_occupations,beta_occupations],dtype=float)
    if state.multiplicity != 1 or np.sum(occ) != state.electron_count:
        raise ValueError('ROKS singlet currently requires an even-electron closed-shell reference')
    c0 = state.coefficients
    if reference_result is not None:
        projected = project_delta_orbitals(reference_result,context)
        c0 = (projected[0] if isinstance(reference_result,ROKSSingletResult)
              else restricted_singlet_seed(state.overlap,projected,occ))
    if orbital_solver == 'least_squares':
        result = run_roks_singlet(state.overlap,c0,occ,callback,**solver_options)
    elif orbital_solver == 'newton':
        from .roks_orbital_response import run_roks_orbital_newton
        result = run_roks_orbital_newton(context,c0,occ,**solver_options)
    elif orbital_solver == 'gmf':
        from .roks_orbital_response import run_roks_gmf
        result = run_roks_gmf(context,c0,occ,**solver_options)
    else:
        raise ValueError('ROKS orbital solver must be least_squares, newton or gmf')
    ground = None if reference_policy=='geometry_only' else run_deltascf(state.overlap,np.array([state.coefficients]*2),
                          np.array([state.alpha_occupations,state.beta_occupations]),callback,
                          selection='aufbau',density_tolerance=1e-10,
                          orbital_tolerance=1e-10,energy_tolerance=1e-12,maximum_iterations=500)
    result = replace(result,reference_state=state,
        core_electrons=np.asarray(state.atomic_numbers)-np.asarray(calculator._valence,dtype=int),
        ground_energy_hartree=None if ground is None else ground.energy_hartree,
        spin_reference_fingerprint=None if spin_reference is None else spin_reference.fingerprint,
        spin_frame_clusters=getattr(spin_reference,'frame_clusters',None))
    if isinstance(reference_result, ROKSSingletResult):
        from matrix_apoc import follow_spin_adapted_singlet
        follow_spin_adapted_singlet(reference_result.to_apoc(),(result.to_apoc(),),
            delta_cross_overlap(reference_result.reference_state,context),
            minimum_overlap=minimum_state_overlap)
    if gradient:
        if not result.converged:
            raise ROKSConvergenceError(result)
        c = result.common_coefficients
        pm, pt = result.densities(), result.densities(triplet=True)
        q = sum(2*(c.T @ f @ c)*o[None,:] for f,o in zip(result.fock_mixed,result.occupations,strict=True))
        q -= sum((c.T @ f @ c)*o[None,:] for f,o in zip(result.fock_triplet,result.triplet_occupations,strict=True))
        weighted = c @ (.5*(q+q.T)) @ c.T
        # Individual mixed/triplet determinants are not stationary. Apply the
        # ONE shared ROKS orthonormality multiplier exactly once.
        zero = np.zeros_like(weighted)
        g = 2*context.analytic_gradient(pm,.5*weighted)-context.analytic_gradient(pt,zero)
        result = replace(result,gradient_hartree_per_bohr=g)
    return result


def spin_adapt_gfn2_singlet(calculator, broken_symmetry_result, *, gradient=True, **options):
    """Reoptimize a retained one-pair Ms=0 determinant into a pure two-determinant singlet CSF.

    This is a restricted spin-adapted variational reoptimization, not exact
    projection of the original unrestricted determinant or its DFT energy.
    """
    from .calculator import ANGSTROM_TO_BOHR
    from types import SimpleNamespace
    state = broken_symmetry_result.reference_state
    if state is None or not broken_symmetry_result.converged:
        raise ValueError('spin adaptation requires a converged molecular BS determinant')
    oa,ob = broken_symmetry_result.occupations
    return evaluate_gfn2_roks_singlet(calculator,state.coordinates_bohr/ANGSTROM_TO_BOHR,
        oa,ob,reference_result=broken_symmetry_result,gradient=gradient,
        ground_state=SimpleNamespace(converged=True,orbital_state=state),**options)


def restricted_singlet_seed(overlap, spin_coefficients, occupations):
    """Gauge-invariant UHF-to-common-orbital seed via corresponding orbitals.

    Occupied MO indices have no identity after an unrestricted eigensolve.
    SVD instead identifies paired and broken-pair subspaces. The closed pairs
    are averaged, the broken pair is symmetrically orthogonalized, and the
    remaining virtual complement is completed in the actual AO metric.
    """
    from scipy.linalg import null_space
    s=np.asarray(overlap);ca,cb=np.asarray(spin_coefficients);oa,ob=np.asarray(occupations)
    closed=np.flatnonzero((oa==1)&(ob==1));alpha=np.flatnonzero((oa==1)&(ob==0));beta=np.flatnonzero((ob==1)&(oa==0))
    if len(alpha)!=1 or len(beta)!=1 or np.sum(oa)!=np.sum(ob):
        raise ValueError('corresponding-orbital singlet seed requires one broken Ms=0 pair')
    a=ca[:,oa>.5];b=cb[:,ob>.5]
    u,values,vh=np.linalg.svd(a.T @ s @ b)
    a=a @ u;b=b @ vh.T
    paired=(a[:,:-1]+b[:,:-1])/np.sqrt(2*(1+values[:-1]))
    open_pair=np.column_stack((a[:,-1],b[:,-1]))
    norm,vectors=np.linalg.eigh(open_pair.T @ s @ open_pair)
    if np.min(norm)<1e-8:
        raise ValueError('collapsed or rank-deficient broken pair cannot seed an open-shell singlet')
    open_pair=open_pair @ ((vectors*norm**-.5) @ vectors.T)
    block=np.column_stack((paired,open_pair))
    eig,rotation=np.linalg.eigh(s)
    half=(rotation*np.sqrt(eig)) @ rotation.T
    inverse=(rotation*eig**-.5) @ rotation.T
    virtual=inverse @ null_space((half @ block).T)
    out=np.empty_like(ca)
    out[:,closed]=paired;out[:,alpha]=open_pair[:,:1];out[:,beta]=open_pair[:,1:]
    out[:,(oa+ob)==0]=virtual
    if not np.allclose(out.T @ s @ out,np.eye(len(s)),atol=1e-8,rtol=0):
        raise ValueError('corresponding-orbital singlet seed lost orthonormality')
    return out
