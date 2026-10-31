"""Checkpointed nuclear Hessians from analytic DeltaSCF/ROKS gradients.

The Hessian is a central finite difference of relaxed, analytically evaluated
first derivatives. It is explicitly numerical, not an analytic CPSCF Hessian.
"""
from __future__ import annotations
from dataclasses import dataclass, fields
from hashlib import sha256
import json
from pathlib import Path
import numpy as np

from .deltascf import DeltaSCFResult, evaluate_gfn2_deltascf
from .roks import ROKSSingletResult, evaluate_gfn2_roks_singlet


@dataclass(frozen=True)
class DeltaSCFHessian:
    reference: DeltaSCFResult | ROKSSingletResult
    hessian_hartree_per_bohr2: np.ndarray
    raw_hessian_hartree_per_bohr2: np.ndarray
    symmetry_error_hartree_per_bohr2: float
    step_bohr: float
    evaluations: int
    reused_points: int
    method: str = 'central_difference_of_analytic_gradients'


def save_derivative_point(path, result, *, request_hash, coordinates_angstrom):
    """Serialize the retained molecular state without pickle or hidden SCF reruns."""
    data = dict(schema='mouse.deltascf.derivative_point.v1',kind=type(result).__name__,
                request_hash=request_hash,coordinates_angstrom=coordinates_angstrom)
    for f in fields(result):
        value = getattr(result,f.name)
        if value is not None and f.name != 'reference_state':
            data['result_'+f.name] = value
    state = result.reference_state
    if state is None:
        raise ValueError('molecular derivative checkpoint requires retained basis metadata')
    for f in fields(state):
        if f.name == 'basis':
            for b in fields(state.basis):
                data['basis_'+b.name] = getattr(state.basis,b.name)
        else:
            data['state_'+f.name] = getattr(state,f.name)
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,**data)
    temporary.replace(path)


def load_derivative_point(path, *, request_hash, coordinates_angstrom, require_gradient=True):
    """Resume a retained point; derivative consumers require gradients by default.

    Energy-only qualification jobs may explicitly set require_gradient=False.
    This preserves converged orbitals without fabricating a missing gradient or
    weakening the default boundary used by nuclear Hessian calculations.
    """
    if not isinstance(require_gradient,(bool,np.bool_)):
        raise ValueError('require_gradient must be a boolean')
    from .electronic_state import GFN2OrbitalState, GFN2OrbitalBasis
    with np.load(path,allow_pickle=False) as data:
        if (str(data['schema']) != 'mouse.deltascf.derivative_point.v1'
                or str(data['request_hash']) != request_hash
                or not np.array_equal(data['coordinates_angstrom'],coordinates_angstrom)):
            raise ValueError('molecular derivative checkpoint does not match this request')
        def value(key):
            a = np.array(data[key],copy=True)
            return a.item() if a.ndim == 0 else a
        basis = GFN2OrbitalBasis(**{f.name:value('basis_'+f.name) for f in fields(GFN2OrbitalBasis)})
        metadata = {f.name:value('state_'+f.name) for f in fields(GFN2OrbitalState) if f.name != 'basis'}
        metadata['ao_shell'] = tuple(metadata['ao_shell'])
        state = GFN2OrbitalState(**metadata,basis=basis)
        kind = str(data['kind'])
        types = {'DeltaSCFResult':DeltaSCFResult,'ROKSSingletResult':ROKSSingletResult}
        if kind not in types:
            raise ValueError('unknown derivative checkpoint state representation')
        cls = types[kind]
        kwargs = {f.name:value('result_'+f.name) for f in fields(cls) if 'result_'+f.name in data}
        result = cls(**kwargs,reference_state=state)
    if (not result.converged
            or (require_gradient and result.gradient_hartree_per_bohr is None)
            or (result.gradient_hartree_per_bohr is not None
                and not np.all(np.isfinite(result.gradient_hartree_per_bohr)))):
        raise ValueError('derivative checkpoint is not a converged gradient point')
    return result


def evaluate_gfn2_deltascf_hessian(calculator, coordinates_angstrom, alpha_occupations,
                                   beta_occupations, *, mode='deltascf', step_bohr=.002,
                                   checkpoint_dir=None, reference_result=None,
                                   spin_reference=None,
                                   **solver_options):
    """Return a state-followed numerical Hessian with per-displacement restart.

    Coordinates are angstrom; displacements and returned derivatives are in
    atomic units. Every displaced solution is followed against the CENTRAL
    state, avoiding path-dependent reference drift. Raw asymmetry is reported
    before symmetrization; it is not hidden by the returned symmetric matrix.
    """
    from .calculator import ANGSTROM_TO_BOHR
    from .parameters import PARAMETER_SHA256
    from .spin_polarization import SPIN_PARAMETER_SHA256
    from .spin_reference import validate_spin_reference
    if reference_result is not None:
        validate_spin_reference(reference_result,spin_reference)
    if mode not in {'deltascf','roks'} or not np.isfinite(step_bohr) or step_bohr <= 0:
        raise ValueError('invalid DeltaSCF Hessian mode or displacement')
    xyz = np.asarray(coordinates_angstrom,dtype=float)
    if xyz.shape != (len(calculator.atomic_numbers),3) or not np.all(np.isfinite(xyz)):
        raise ValueError('invalid Hessian geometry')
    # Native model variants must not collide in the on-disk cache.
    request = dict(mode=mode,step_bohr=float(step_bohr),xyz=xyz.tolist(),
        atoms=calculator.atomic_numbers,charge=calculator.charge,multiplicity=calculator.multiplicity,
        alpha=np.asarray(alpha_occupations).tolist(),beta=np.asarray(beta_occupations).tolist(),
        solver=solver_options,model_parameters=PARAMETER_SHA256,spin_parameters=SPIN_PARAMETER_SHA256,
        calculator_options=repr(calculator._hessian_process_options()),
        spin_reference_fingerprint=None if spin_reference is None else spin_reference.fingerprint,
        reference_hash=(None if reference_result is None else sha256(np.asarray(reference_result.coefficients).tobytes()).hexdigest()),
        source_hash=sha256(b''.join(Path(__file__).with_name(f).read_bytes()
            for f in ('deltascf.py','roks.py','spin_polarization.py','spin_reference.py','tensor_spin.py','deltascf_derivatives.py'))).hexdigest())
    digest = sha256(json.dumps(request,sort_keys=True,default=str).encode()).hexdigest()
    cache = None if checkpoint_dir is None else Path(checkpoint_dir)/digest
    if cache is not None:
        cache.mkdir(parents=True,exist_ok=True)
        (cache/'request.json').write_text(json.dumps(request,indent=2,default=str)+'\n')
    evaluator = evaluate_gfn2_deltascf if mode == 'deltascf' else evaluate_gfn2_roks_singlet
    evaluated,reused = 0,0
    def point(label,geometry,reference):
        nonlocal evaluated,reused
        path = None if cache is None else cache/(label+'.npz')
        if path is not None and path.exists():
            result = load_derivative_point(path,request_hash=digest,coordinates_angstrom=geometry)
            reused += 1
        else:
            result = evaluator(calculator,geometry,alpha_occupations,beta_occupations,
                gradient=True,reference_result=reference,spin_reference=spin_reference,**solver_options)
            evaluated += 1
            if path is not None:
                save_derivative_point(path,result,request_hash=digest,coordinates_angstrom=geometry)
        return result
    same_center = (reference_result is not None and reference_result.converged
        and reference_result.gradient_hartree_per_bohr is not None
        and isinstance(reference_result,ROKSSingletResult) == (mode == 'roks')
        and np.array_equal(reference_result.occupations,[alpha_occupations,beta_occupations])
        and np.allclose(reference_result.reference_state.coordinates_bohr,
                        xyz*ANGSTROM_TO_BOHR,atol=1e-10,rtol=0))
    if same_center:
        central = reference_result
        reused += 1
        if cache is not None and not (cache/'central.npz').exists():
            save_derivative_point(cache/'central.npz',central,request_hash=digest,coordinates_angstrom=xyz)
    else:
        central = point('central',xyz,reference_result)
    raw = np.empty((xyz.size,xyz.size))
    for coordinate in range(xyz.size):
        delta = np.zeros(xyz.size);delta[coordinate] = step_bohr/ANGSTROM_TO_BOHR
        displaced = []
        for sign in (-1,1):
            result = point(f'{coordinate:04d}_{sign:+d}',xyz+sign*delta.reshape(xyz.shape),central)
            displaced.append(result.gradient_hartree_per_bohr.reshape(-1))
        raw[:,coordinate] = (displaced[1]-displaced[0])/(2*step_bohr)
    asymmetry = float(np.max(np.abs(raw-raw.T)))
    return DeltaSCFHessian(central,.5*(raw+raw.T),raw,asymmetry,float(step_bohr),evaluated,reused)
