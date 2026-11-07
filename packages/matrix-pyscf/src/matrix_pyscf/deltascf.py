"""Fixed-occupation unrestricted OO-DFT using analytic PySCF response.

The optimizer solves orbital stationarity, retaining negative curvature. It
never diagonalizes the Fock matrix to reassign occupations. The separate ROKS adapter constructs common-orbital singlets.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


def _fingerprint(mf, occupations):
    mol=mf.mol; grids=getattr(mf, 'grids', None)
    payload = dict(atoms=mol.atom_charges().tolist(), coordinates_bohr=mol.atom_coords().tolist(),
                   basis=mol._basis, ecp=mol._ecp, cart=mol.cart,
                   charge=mol.charge, spin=mol.spin, xc=getattr(mf, 'xc', 'HF'),
                   density_fit=hasattr(mf, 'with_df'),
                   auxbasis=getattr(getattr(mf, 'with_df', None), 'auxbasis', None),
                   grids=None if grids is None else dict(level=grids.level,atom_grid=grids.atom_grid,
                       prune=getattr(grids.prune,'__qualname__',str(grids.prune))),
                   occupations=np.asarray(occupations).tolist())
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=lambda a:np.asarray(a).tolist()).encode()).hexdigest()


def _density(coefficients, occupations):
    return np.array([(c * o) @ c.T for c, o in zip(coefficients, occupations)])


def _overlap(reference, coefficients, occupations, metric):
    value = 1.
    for ref, c, occ in zip(reference, coefficients, occupations):
        occupied = occ > .5
        value *= abs(float(np.linalg.det(ref[:, occupied].T @ metric @ c[:, occupied])))
    return value


def _rotate(coefficients, occupations, step):
    from scipy.linalg import expm
    result = []; offset = 0
    for c, occ in zip(coefficients, occupations):
        occupied = np.flatnonzero(occ > .5); virtual = np.flatnonzero(occ < .5)
        count = len(occupied)*len(virtual)
        block = step[offset:offset+count].reshape(len(virtual), len(occupied))
        kappa = np.zeros((c.shape[1], c.shape[1]))
        kappa[np.ix_(virtual, occupied)] = block
        kappa[np.ix_(occupied, virtual)] = -block.T
        result.append(c @ expm(kappa)); offset += count
    if offset != len(step):
        raise ValueError('orbital step has incompatible dimension')
    return np.array(result)


@dataclass(frozen=True)
class PySCFDeltaSCFResult:
    coefficients: np.ndarray
    occupations: np.ndarray
    reference_coefficients: np.ndarray
    energy_hartree: float
    ground_energy_hartree: float | None
    orbital_residual: float
    converged: bool
    iterations: int
    reference_overlap: float
    spin_squared: float
    gradient_hartree_per_bohr: np.ndarray | None
    model_fingerprint: str
    history: tuple[dict, ...]
    hessian_hartree_per_bohr2: np.ndarray | None = None
    response_certificate: dict | None = None

    @property
    def excitation_energy_hartree(self):
        return None if self.ground_energy_hartree is None else self.energy_hartree-self.ground_energy_hartree

    def save_checkpoint(self, path):
        """Atomically preserve the accepted state and immutable target orbitals."""
        path = Path(path); temporary = path.with_name(path.name+'.tmp')
        metadata = dict(fingerprint=self.model_fingerprint, energy_hartree=self.energy_hartree,
                        ground_energy_hartree=self.ground_energy_hartree, history=self.history)
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, coefficients=self.coefficients, occupations=self.occupations,
                                reference_coefficients=self.reference_coefficients,
                                metadata=json.dumps(metadata))
        temporary.replace(path)

    def to_apoc(self, mean_field, state_id=1):
        from matrix_apoc import DeterminantOrbitals, DeltaSCFState
        if _fingerprint(mean_field, self.occupations) != self.model_fingerprint:
            raise ValueError('APOC export molecular model does not match the optimized determinant')
        if mean_field.mol.has_ecp():
            raise NotImplementedError('APOC ECP core metadata is not yet supported by the DFT adapter')
        return DeltaSCFState(state_id, DeterminantOrbitals(
            mean_field.mol.atom_charges(), mean_field.mol.atom_coords(), mean_field.get_ovlp(),
            tuple(self.coefficients), tuple(self.occupations),
            np.zeros(mean_field.mol.natm, dtype=int), charge=mean_field.mol.charge,
            multiplicity=mean_field.mol.spin+1, total_energy_hartree=self.energy_hartree,
            model='pyscf.oo-dft.'+self.model_fingerprint))


class PySCFDeltaSCFConvergenceError(RuntimeError):
    def __init__(self, result):
        self.result = result
        super().__init__(f'OO-DFT stationarity not converged: residual {result.orbital_residual:.3e}')


def optimize_deltascf(mean_field, coefficients, occupations, *, ground_energy_hartree=None,
                      reference_coefficients=None, restart=None, checkpoint=None,
                      maximum_iterations=80, orbital_tolerance=1e-7, maximum_step=.2,
                      minimum_reference_overlap=.1, include_gradient=False, include_hessian=False, strict=True):
    """Find a stationary determinant using the ground-state UHF/UKS backend.

    ``coefficients`` has shape (2, nao, nmo); explicit binary spin occupations
    refer to its columns. Occupations and the target reference remain fixed.
    A Newton--Krylov step uses the full analytic orbital Hessian action, including
    the XC kernel. A residual-norm line search accepts saddle steps without
    requiring energy decrease. No finite differences or TDA calls are used.

    Only a fresh molecular UHF/UKS object without solvent, NLC, smearing or
    symmetry constraints is supported. Supply a separately converged ground
    energy when an excitation energy is wanted. Nuclear gradients are exposed
    only after orbital convergence. ``include_hessian`` adds the analytic coupled
    nuclear response for full-integral HF and LDA/GGA (including hybrid) DFT.
    """
    from pyscf import scf
    from pyscf.soscf.newton_ah import gen_g_hop_uhf
    from scipy.sparse.linalg import LinearOperator, gmres
    mf = mean_field
    if not isinstance(mf, scf.uhf.UHF) or mf.mol.symmetry:
        raise ValueError('OO-DFT requires an unrestricted molecular mean field without symmetry')
    if (getattr(mf, 'with_solvent', None) is not None or getattr(mf, 'nlc', '')
            or getattr(mf, 'sigma', 0) or getattr(mf, 'with_x2c', None) is not None):
        raise NotImplementedError('solvent, NLC, smearing and relativistic OO-DFT response are not qualified')
    if include_hessian:
        from ._oo_nuclear import validate_hessian_model
        validate_hessian_model(mf)
        orbital_tolerance=min(orbital_tolerance,1e-10)
    c = np.asarray(coefficients); occ = np.asarray(occupations)
    if np.iscomplexobj(c):
        raise ValueError('OO-DFT currently requires real orbitals')
    c = np.array(c, dtype=float, copy=True); occ = np.array(occ, dtype=float, copy=True)
    if (c.ndim != 3 or c.shape[0] != 2 or c.shape[1] != mf.mol.nao_nr()
            or occ.shape != (2, c.shape[2]) or not np.all(np.isfinite(c))
            or not np.all((occ == 0) | (occ == 1))
            or not np.array_equal(occ.sum(axis=1), mf.mol.nelec)
            or maximum_iterations < 1 or orbital_tolerance <= 0 or maximum_step <= 0
            or not 0 <= minimum_reference_overlap <= 1):
        raise ValueError('invalid OO-DFT orbitals, spin occupations or solver settings')
    fingerprint = _fingerprint(mf, occ)
    if reference_coefficients is not None and np.iscomplexobj(reference_coefficients):
        raise ValueError('OO-DFT currently requires real reference orbitals')
    reference = c.copy() if reference_coefficients is None else np.array(reference_coefficients, dtype=float, copy=True)
    history = []
    if restart is not None:
        with np.load(restart, allow_pickle=False) as data:
            metadata = json.loads(str(data['metadata']))
            if metadata['fingerprint'] != fingerprint or not np.array_equal(data['occupations'], occ):
                raise ValueError('OO-DFT restart model/occupations mismatch')
            c = data['coefficients'].copy(); reference = data['reference_coefficients'].copy()
            if ground_energy_hartree is None:ground_energy_hartree=metadata['ground_energy_hartree']
    metric = mf.get_ovlp()
    for orbitals in (c, reference):
        if (orbitals.shape != c.shape or not np.all(np.isfinite(orbitals))
                or any(not np.allclose(x.T@metric@x, np.eye(c.shape[2]), atol=1e-8, rtol=0) for x in orbitals)):
            raise ValueError('OO-DFT orbitals must be AO-metric orthonormal')
    if getattr(mf, 'xc', None):
        if mf._numint.libxc.is_nlc(mf.xc):
            raise NotImplementedError('nonlocal correlation response is not supported')
    # Work on a copy so the caller retains its ground state and occupations.
    mf = mf.copy(); mf.mo_occ = occ.copy(); mf.mo_coeff = c.copy()
    converged = False; gradient = None
    def evaluate(orbitals):
        dm = _density(orbitals, occ)
        potential = mf.get_veff(mf.mol, dm)
        fock = np.asarray(mf.get_hcore())+potential
        energy = float(mf.energy_tot(dm=dm, vhf=potential))
        g, action, diagonal = gen_g_hop_uhf(mf, orbitals, occ, fock_ao=fock, with_symmetry=False)
        return energy, np.asarray(g), action, diagonal, fock
    for iteration in range(maximum_iterations+1):
        energy, g, action, diagonal, fock = evaluate(c)
        residual = float(np.max(abs(g))) if len(g) else 0.
        identity = _overlap(reference, c, occ, metric)
        if identity < minimum_reference_overlap:
            raise ValueError('OO-DFT immutable target-overlap guard rejected the determinant')
        history.append(dict(iteration=iteration, energy_hartree=energy, orbital_residual=residual,
                            reference_overlap=identity))
        result = PySCFDeltaSCFResult(c.copy(), occ.copy(), reference.copy(), energy,
            ground_energy_hartree, residual, residual < orbital_tolerance, iteration,
            identity, float(scf.uhf.spin_square((c[0][:,occ[0]>.5], c[1][:,occ[1]>.5]), metric)[0]),
            None, fingerprint, tuple(history))
        if checkpoint is not None:result.save_checkpoint(checkpoint)
        if result.converged:converged=True; break
        if iteration == maximum_iterations:break
        operator = LinearOperator((len(g),len(g)), matvec=action, dtype=float)
        denominator = np.where(abs(diagonal)<1e-4, np.where(diagonal<0,-1e-4,1e-4), diagonal)
        preconditioner = LinearOperator(operator.shape, matvec=lambda x:x/denominator, dtype=float)
        step, info = gmres(operator, -g, M=preconditioner, rtol=1e-4, atol=0.,
                           restart=min(60,len(g)), maxiter=8)
        if not np.all(np.isfinite(step)):step=np.zeros_like(g)
        # A residual-descent fallback remains valid for indefinite Hessians.
        candidates = [step, -action(g)]
        accepted = False
        for direction in candidates:
            norm = np.linalg.norm(direction)
            if norm < 1e-16:continue
            direction = direction*min(1., maximum_step/norm)
            for fraction in (1., .5, .25, .125, .0625, .03125):
                trial = _rotate(c, occ, fraction*direction)
                if _overlap(reference,trial,occ,metric)<minimum_reference_overlap:continue
                _, trial_g, _, _, _ = evaluate(trial)
                if np.linalg.norm(trial_g) < (1-1e-4*fraction)*np.linalg.norm(g):
                    c=trial; accepted=True; break
            if accepted:break
        if not accepted:break
    if converged and include_gradient:
        from scipy.linalg import eigh
        canonical=c.copy(); mo_energy=np.empty(occ.shape)
        for spin in range(2):
            for selected in (occ[spin]>.5, occ[spin]<.5):
                ids=np.flatnonzero(selected)
                if not len(ids):continue
                block=c[spin][:,ids]
                values,rotation=eigh(block.T@fock[spin]@block)
                canonical[spin][:,ids]=block@rotation; mo_energy[spin,ids]=values
        derivative=mf.nuc_grad_method()
        if hasattr(derivative, 'grid_response'):derivative.grid_response=True
        gradient=np.asarray(derivative.kernel(mo_energy=mo_energy,mo_coeff=canonical,mo_occ=occ))
        from dataclasses import replace
        result=replace(result,gradient_hartree_per_bohr=gradient)
    if converged and include_hessian:
        from dataclasses import replace
        from ._oo_surface import OrbitalSurface
        from ._oo_nuclear import nuclear_response
        if getattr(mf,'with_df',None) is not None:
            raise NotImplementedError('density-fitted OO nuclear Hessian not qualified')
        gradient,hessian,certificate=nuclear_response(OrbitalSurface(mf,c,occ),c,hessian=True)
        result=replace(result,gradient_hartree_per_bohr=gradient,hessian_hartree_per_bohr2=hessian,response_certificate=certificate)
    if strict and not converged:raise PySCFDeltaSCFConvergenceError(result)
    return result
