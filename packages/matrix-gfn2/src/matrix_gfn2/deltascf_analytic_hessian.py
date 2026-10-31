"""Analytic DeltaSCF/ROKS Hessians from constrained orbital response.

Native Gaussian integral recurrences and the existing complete GFN2 derivative
kernels provide E_RR and F_R at fixed AO density. Analytic spin derivatives,
orthonormality derivatives and the coupled orbital Schur complement complete
the relaxed Hessian. No nuclear displacement or finite difference is used.
This dense molecular reference path deliberately exposes singular responses.
"""
from __future__ import annotations
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np

from .deltascf import evaluate_gfn2_deltascf, prepare_gfn2_deltascf
from .roks import ROKSSingletResult, evaluate_gfn2_roks_singlet


@dataclass(frozen=True)
class AnalyticDeltaSCFHessian:
    reference: object
    hessian_hartree_per_bohr2: np.ndarray
    frozen_orbital_hessian_hartree_per_bohr2: np.ndarray
    response_hessian_hartree_per_bohr2: np.ndarray
    orbital_hessian_hartree: np.ndarray
    orbital_nuclear_coupling: np.ndarray
    orbital_response: np.ndarray
    maximum_response_residual: float
    raw_symmetry_error_hartree_per_bohr2: float
    minimum_absolute_orbital_curvature_hartree: float | None
    method: str = 'analytic_constrained_orbital_schur_complement'


def _density(c,o):
    return (c*o) @ c.T


def _density_first(c,o,dc):
    value = (dc*o) @ c.T
    return value+value.T


def _density_second(c,o,da,db,dab):
    value = (dab*o) @ c.T + (da*o) @ db.T
    return value+value.T


def _fixed_density_geometry(context, spin_density):
    """Analytic fixed-P nuclear blocks, independent of orbital canonicality.

    The temporary density reference is a derivative container, not a ground
    state. Its natural occupations reconstruct P exactly; zero orbital-energy
    placeholders request the fixed-P E_RR block without any Pulay multiplier.
    All true orthonormality contributions are assembled below in the metric
    gauge, so none are omitted or counted twice.
    """
    from matrix_apoc import natural_orbitals
    from .molecular_second_response import _integrals
    calc,state = context.calculator,context.state
    p = np.sum(spin_density,axis=0)
    natural = natural_orbitals(p,state.overlap)
    reference = SimpleNamespace(**vars(state),nao=state.nao,nmo=state.nmo)
    reference.coefficients = natural.coefficients
    reference.alpha_occupations = .5*natural.occupations
    reference.beta_occupations = .5*natural.occupations
    reference.orbital_energies_hartree = np.zeros(state.nmo)
    fixed = calc._prepare_density_response_gradient_context(reference)
    dimension = state.coordinates_bohr.size
    ds = np.empty((dimension,state.nao,state.nao))
    df = np.empty_like(ds)
    second_s = np.empty((dimension,dimension,state.nao,state.nao))
    explicit = np.empty((dimension,dimension))
    for b in range(dimension):
        v = np.zeros(dimension);v[b] = 1.
        direction = v.reshape(-1,3)
        ds[b],df[b] = calc._ground_state_directional_derivative_matrices(reference,direction,context=fixed)
        overlap_jet,_,_ = _integrals(calc,reference,fixed,direction)
        second_s[:,b] = overlap_jet.ab
        explicit[:,b] = calc.ground_state_explicit_pair_hvp(reference,direction,_context=fixed).total.ravel()
    return fixed['kernel'],ds,df,second_s,explicit


def _spin_geometry(polarization,spin_density,ds,d2s):
    """Exact fixed-P first Fock and second energy derivatives of shell spin."""
    shells = polarization.orbital_shell
    w,s = polarization.coupling_hartree,polarization.overlap
    magnet = spin_density[0]-spin_density[1]
    def moments(metric):
        return np.bincount(shells,weights=np.sum(magnet*metric,axis=1),minlength=len(w))
    m = moments(s);dm = np.array([moments(x) for x in ds])
    v = (w @ m)[shells]
    dv = (dm @ w.T)[:,shells]
    dpotential = .5*(ds*(v[:,None]+v[None,:])
                       + s[None]*(dv[:,:,None]+dv[:,None,:]))
    h = dm @ w @ dm.T
    for a in range(len(ds)):
        for b in range(len(ds)):
            h[a,b] += float(m @ w @ moments(d2s[a,b]))
    if hasattr(polarization,'tensor_geometry'):
        df,rr=polarization.tensor_geometry(spin_density,ds,d2s)
        dpotential=dpotential+df;h=h+rr
    return np.array([dpotential,-dpotential]).transpose(1,0,2,3),h


def _response_blocks(coefficients, components, ds, d2s, explicit_components, kernels, polarization, *, orbital_pairs=None):
    """Analytic Hessian in a local symmetric-metric/exponential-orbital chart."""
    n = coefficients[0].shape[0];nr = len(ds)
    shared = len(coefficients) == 1
    cs = coefficients*2 if shared else coefficients
    # Components are (multiplet weight, spin occupations, spin Fock matrices).
    generators = []
    for sector in range(1 if shared else 2):
        left,right = np.triu_indices(n,1)
        if orbital_pairs is not None:
            if not shared:raise ValueError('explicit orbital pairs require a shared spatial chart')
            left,right=np.asarray(orbital_pairs,dtype=int)
            active=np.ones(len(left),dtype=bool)
        elif shared:
            labels = np.vstack([o for _,o,_ in components])
            active = np.any(labels[:,left] != labels[:,right],axis=0)
        else:
            labels = components[0][1][sector]
            active = labels[left] != labels[right]
        for i,j in zip(left[active],right[active]):
            g = np.zeros((n,n));g[i,j] = 1.;g[j,i] = -1.
            generators.append((sector,g))
    nk = len(generators)
    def affected(sector,spin):
        return shared or sector == spin
    # Coefficient derivatives; chart C(R,k)=C0 (C0.T S(R) C0)^-1/2 exp(k).
    metric = [np.array([c.T @ d @ c for d in ds]) for c in cs]
    cr = [np.array([-.5*c @ m for m in mm]) for c,mm in zip(cs,metric)]
    ck = [[c @ g if affected(sector,spin) else np.zeros_like(c)
           for sector,g in generators] for spin,c in enumerate(cs)]
    hrr = np.zeros((nr,nr));hkk = np.zeros((nk,nk));hkr = np.zeros((nk,nr))
    for index,(weight,occ,fock) in enumerate(components):
        ps = np.array([_density(c,o) for c,o in zip(cs,occ)])
        kernel = kernels[index]
        spin_fr,spin_rr = _spin_geometry(polarization,ps,ds,d2s)
        f_explicit = explicit_components[index][0][:,None]+spin_fr
        explicit_rr = explicit_components[index][1]+spin_rr
        pr = np.array([[_density_first(cs[s],occ[s],cr[s][r]) for s in range(2)] for r in range(nr)])
        pk = np.array([[_density_first(cs[s],occ[s],ck[s][k]) for s in range(2)] for k in range(nk)])
        def density_kernel(dp):
            common = kernel(np.sum(dp,axis=0))
            spin_fock,_,_ = polarization.evaluate(dp)
            return common[None]+spin_fock
        kr = np.array([density_kernel(p) for p in pr])
        kk = np.array([density_kernel(p) for p in pk])
        for a in range(nr):
            for b in range(nr):
                value = explicit_rr[a,b] + np.sum(f_explicit[a]*pr[b])+np.sum(f_explicit[b]*pr[a])+np.sum(pr[a]*kr[b])
                for spin,c in enumerate(cs):
                    ma,mb = metric[spin][a],metric[spin][b]
                    second_c = c @ (.375*(ma @ mb+mb @ ma)-.5*(c.T @ d2s[a,b] @ c))
                    second_p = _density_second(c,occ[spin],cr[spin][a],cr[spin][b],second_c)
                    value += np.sum(fock[spin]*second_p)
                hrr[a,b] += weight*value
        # Contract all density-kernel bilinears at once. The previous literal
        # pair loop formed dense second coefficient derivatives for every
        # orbital pair, needlessly costing O(nk**2*n**3).
        orbital_block = pk.reshape(nk,2*n*n) @ kk.reshape(nk,2*n*n).T
        one_body = np.zeros_like(orbital_block)
        for spin,c in enumerate(cs):
            indices = [a for a,(sector,_) in enumerate(generators) if affected(sector,spin)]
            pairs = [np.argwhere(generators[a][1] == 1.)[0] for a in indices]
            if not pairs:
                continue
            il,jl = np.asarray(pairs).T
            factors = 2*(occ[spin,jl]-occ[spin,il])
            fmo = c.T @ fock[spin] @ c
            moving = np.array([g.T @ fmo+fmo @ g for _,g in (generators[a] for a in indices)])
            one_body[np.ix_(indices,indices)] += (moving[:,il,jl]*factors).T
            nuclear_fmo = (cr[spin].transpose(0,2,1) @ fock[spin] @ c
                + c.T @ fock[spin] @ cr[spin]
                + c.T @ (f_explicit[:,spin]+kr[:,spin]) @ c)
            hkr[indices] += weight*(nuclear_fmo[:,il,jl]*factors).T
        # Symmetric local exponential-chart curvature is the symmetric part
        # of the moving-frame residual derivative, including off stationarity.
        hkk += weight*(orbital_block+.5*(one_body+one_body.T))
    return hrr,hkk,hkr


def evaluate_gfn2_deltascf_analytic_hessian(calculator,coordinates_angstrom,
    alpha_occupations,beta_occupations,*,mode='deltascf',reference_result=None,
    response_singularity_tolerance=1e-8,spin_reference=None,**solver_options):
    """Dense analytic relaxed Hessian for a determinant or two-shell S=0 ROKS CSF."""
    if not np.isfinite(response_singularity_tolerance) or response_singularity_tolerance <= 0:
        raise ValueError('orbital response singularity tolerance must be finite and positive')
    if mode not in {'deltascf','roks'}:
        raise ValueError('analytic Hessian mode must be deltascf or roks')
    evaluator = evaluate_gfn2_deltascf if mode == 'deltascf' else evaluate_gfn2_roks_singlet
    result = reference_result
    if result is None:
        solver_options.setdefault('orbital_tolerance',1e-10)
        if mode == 'deltascf':
            solver_options.setdefault('density_tolerance',1e-10)
            solver_options.setdefault('energy_tolerance',1e-12)
            solver_options.setdefault('maximum_iterations',1000)
        result = evaluator(calculator,coordinates_angstrom,alpha_occupations,beta_occupations,
                            gradient=True,spin_reference=spin_reference,**solver_options)
    from .spin_reference import validate_spin_reference
    validate_spin_reference(result,spin_reference)
    from .calculator import ANGSTROM_TO_BOHR
    if (not result.converged or result.orbital_residual > 1e-8
        or not np.allclose(result.reference_state.coordinates_bohr,
                           np.asarray(coordinates_angstrom)*ANGSTROM_TO_BOHR,atol=1e-10,rtol=0)
        or not np.array_equal(result.occupations,[alpha_occupations,beta_occupations])
        or isinstance(result,ROKSSingletResult) != (mode == 'roks')):
        raise ValueError('analytic Hessian requires a matching, tightly stationary molecular reference')
    reference_options=(dict(orbital_reference=result.reference_state)
        if isinstance(result,ROKSSingletResult) and result.ground_energy_hartree is None
        else dict(ground_state=SimpleNamespace(converged=True,orbital_state=result.reference_state)))
    context = prepare_gfn2_deltascf(calculator,coordinates_angstrom,
        spin_reference=spin_reference,**reference_options)
    if getattr(context.polarization,'vertical_only',False):
        raise NotImplementedError('vertical-only response prototypes do not provide nuclear Hessians')
    if mode == 'deltascf':
        coefficients = list(result.coefficients)
        components = [(1.,result.occupations,result.fock)]
    else:
        coefficients = [result.common_coefficients]
        components = [(2.,result.occupations,result.fock_mixed),
                      (-1.,result.triplet_occupations,result.fock_triplet)]
    kernels=[];explicit=[]
    for _,occ,_ in components:
        cs = coefficients*2 if len(coefficients)==1 else coefficients
        spin_density = np.array([_density(c,o) for c,o in zip(cs,occ)])
        kernel,ds,df,d2s,err = _fixed_density_geometry(context,spin_density)
        if context.charge_correction is not None:
            charge=context.charge_correction
            dfield,efield=charge.geometry(np.sum(spin_density,axis=0),ds,d2s)
            df=df+dfield;err=err+efield
            base=kernel
            kernel=lambda dp,base=base:base(dp)+charge.response(dp)
        kernels.append(kernel);explicit.append((df,err))
    hrr,a,b = _response_blocks(coefficients,components,ds,d2s,explicit,kernels,context.polarization)
    a = .5*(a+a.T)
    curvature = np.linalg.eigvalsh(a)
    smallest = float(np.min(np.abs(curvature))) if curvature.size else None
    if smallest is not None and smallest < response_singularity_tolerance:
        raise ValueError(f'orbital response is singular: min |curvature|={smallest:.3e} hartree')
    response = np.linalg.solve(a,-b)
    residual = float(np.max(np.abs(a @ response+b),initial=0.))
    correction = b.T @ response
    total = hrr+correction
    asymmetry = float(np.max(np.abs(total-total.T)))
    return AnalyticDeltaSCFHessian(result,.5*(total+total.T),hrr,correction,a,b,response,
                                  residual,asymmetry,smallest)
