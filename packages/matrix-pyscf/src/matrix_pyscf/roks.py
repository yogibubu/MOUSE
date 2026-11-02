"""Spin-adapted two-open-orbital ROKS with a shared spatial orbital set.

The singlet CSF is spin-pure. Its 2 E_mixed - E_triplet functional energy is an
approximation, not exact projection of independently relaxed UKS energies.
"""
from dataclasses import dataclass,replace
import hashlib,json
from pathlib import Path
import numpy as np
from .deltascf import _fingerprint


@dataclass(frozen=True)
class PySCFROKSResult:
    common_coefficients: np.ndarray
    occupations: np.ndarray
    reference_coefficients: np.ndarray
    energy_hartree: float
    ground_energy_hartree: float|None
    orbital_residual: float
    converged: bool
    iterations: int
    reference_overlap: float
    model_fingerprint: str
    history: tuple
    overlap: np.ndarray
    gradient_hartree_per_bohr: np.ndarray|None=None
    hessian_hartree_per_bohr2: np.ndarray|None=None
    response_certificate: dict|None=None
    solver_certificate: dict|None=None

    def __post_init__(self):
        c=np.asarray(self.common_coefficients);o=np.asarray(self.occupations);s=np.asarray(self.overlap)
        if (c.ndim!=2 or c.shape!=s.shape or o.shape!=(2,len(c))
            or not np.all(np.isfinite(c)) or not np.all(np.isfinite(s))
            or not np.all((o==0)|(o==1)) or np.count_nonzero(o[0]-o[1]==1)!=1
            or np.count_nonzero(o[1]-o[0]==1)!=1
            or not np.allclose(c.T@s@c,np.eye(len(c)),atol=1e-8,rtol=0)):
            raise ValueError('invalid spin-adapted common-orbital result')
        for name in ('common_coefficients','occupations','reference_coefficients','overlap',
                     'gradient_hartree_per_bohr','hessian_hartree_per_bohr2'):
            value=getattr(self,name)
            if value is not None:
                value=np.array(value,copy=True);value.setflags(write=False);object.__setattr__(self,name,value)

    @property
    def spin_squared(self):return 0.

    @property
    def excitation_energy_hartree(self):
        return None if self.ground_energy_hartree is None else self.energy_hartree-self.ground_energy_hartree

    def save_checkpoint(self,path):
        path=Path(path);temporary=path.with_name(path.name+'.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream,coefficients=self.common_coefficients,occupations=self.occupations,
                reference_coefficients=self.reference_coefficients,metadata=json.dumps(dict(fingerprint=self.model_fingerprint,
                    ground_energy_hartree=self.ground_energy_hartree,energy_hartree=self.energy_hartree,
                    solver_certificate=self.solver_certificate,history=self.history)))
        temporary.replace(path)

    def to_apoc(self,mean_field,state_id=1):
        from matrix_apoc import DeterminantOrbitals,DeltaSCFState,SpinAdaptedSinglet
        if _model(mean_field,self.occupations)!=self.model_fingerprint:raise ValueError('ROKS APOC model mismatch')
        if mean_field.mol.has_ecp():raise NotImplementedError('ROKS ECP export is not qualified')
        c=self.common_coefficients
        mixed=DeltaSCFState(state_id,DeterminantOrbitals(mean_field.mol.atom_charges(),mean_field.mol.atom_coords(),
            mean_field.get_ovlp(),(c,c),tuple(self.occupations),np.zeros(mean_field.mol.natm,dtype=int),
            charge=mean_field.mol.charge,multiplicity=1,total_energy_hartree=self.energy_hartree,
            model='pyscf.roks.'+self.model_fingerprint))
        return SpinAdaptedSinglet(state_id,mixed,self.energy_hartree)


def _singlet_overlap(reference,c,occ,metric):
    a,b=[reference[:,o>.5] for o in occ];x,y=[c[:,o>.5] for o in occ]
    return abs(float(np.linalg.det(a.T@metric@x)*np.linalg.det(b.T@metric@y)+
                     np.linalg.det(a.T@metric@y)*np.linalg.det(b.T@metric@x)))


def _model(mf,occ):return hashlib.sha256(('common-orbital multiplet-sum ROKS|'+_fingerprint(mf,occ)).encode()).hexdigest()


class PySCFROKSConvergenceError(RuntimeError):
    def __init__(self,result):
        self.result=result
        super().__init__(f'ROKS not certified: residual {result.orbital_residual:.3e}; '
                         f"status={(result.solver_certificate or {}).get('status','not converged')}")


def optimize_roks(mean_field,coefficients,occupations,*,ground_energy_hartree=None,
                  restart=None,checkpoint=None,maximum_iterations=80,orbital_tolerance=1e-8,
                  maximum_step=.2,minimum_reference_overlap=.1,include_gradient=False,
                  include_hessian=False,strict=True,orbital_solver='trust',
                  initial_trust_radius=None,minimum_step_overlap=.8,
                  certify_index=False,target_saddle_index=None,index_threshold=1e-6,
                  reference_coefficients=None,freeze_and_release=True,initialize_closed_orbitals=False):
    """Optimize 2 E_M - E_T with common orbitals; construct its singlet CSF.

    Input is a full real AO-metric orthonormal spatial orbital matrix and two
    binary occupations with one alpha-only and one beta-only orbital. The
    input mean field is molecular UHF/UKS, Ms=0, without symmetry, solvent or
    density fitting. Numerical parameters are unchanged across the components.

    The default residual trust solver uses the exact moving-frame Jacobian and
    its adjoint. ``reference_coefficients`` retains a target independently of
    the initial guess; a restart must match this target. Recovery temporarily
    freezes the two open orbitals, then releases every nonredundant rotation.
    It changes initialization only, never the energy or the final response.
    ``initialize_closed_orbitals`` applies this preparation before the first
    search, without replacing an accepted restart.
    ``certify_index`` counts negative stationary orbital modes; specifying
    ``target_saddle_index`` additionally rejects a mismatched root. Neither
    curvature nor overlap alone establishes the spectroscopic state identity.
    Accepted checkpoints include solver diagnostics and iteration history.
    ``orbital_solver='newton'`` retains a clipped Newton comparison path.
    """
    from pyscf import scf
    from ._oo_surface import OrbitalSurface
    from ._roks_optimizer import solve,curvature_certificate,relax_closed_orbitals
    mf=mean_field;c=np.asarray(coefficients);occ=np.asarray(occupations)
    if (not isinstance(mf,scf.uhf.UHF) or mf.mol.spin!=0 or mf.mol.symmetry
        or np.iscomplexobj(c) or c.shape!=(mf.mol.nao_nr(),mf.mol.nao_nr())
        or occ.shape!=(2,c.shape[1]) or not np.all((occ==0)|(occ==1))
        or not np.array_equal(occ.sum(axis=1),mf.mol.nelec)
        or np.count_nonzero(occ[0]-occ[1]==1)!=1 or np.count_nonzero(occ[1]-occ[0]==1)!=1
        or maximum_iterations<1 or orbital_tolerance<=0 or maximum_step<=0
        or not 0<=minimum_reference_overlap<=1 or not 0<=minimum_step_overlap<=1
        or orbital_solver not in ('trust','newton') or index_threshold<=0
        or (initial_trust_radius is not None and (not np.isfinite(initial_trust_radius) or initial_trust_radius<=0))
        or (target_saddle_index is not None and (isinstance(target_saddle_index,(bool,np.bool_))
            or not isinstance(target_saddle_index,(int,np.integer)) or target_saddle_index<0))):raise ValueError('invalid common-orbital ROKS model or occupations')
    if (getattr(mf,'with_df',None) is not None or getattr(mf,'with_solvent',None) is not None
        or getattr(mf,'with_x2c',None) is not None or getattr(mf,'sigma',0) or getattr(mf,'disp',None)
        or getattr(mf,'nlc','')):raise NotImplementedError('ROKS optional response model is not qualified')
    if getattr(mf,'xc',None) and mf._numint.libxc.is_nlc(mf.xc):raise NotImplementedError('ROKS NLC response unsupported')
    if include_hessian:
        from ._oo_nuclear import validate_hessian_model
        validate_hessian_model(mf)
        orbital_tolerance=min(orbital_tolerance,1e-10)
    identity=_model(mf,occ);c=np.array(c,dtype=float,copy=True)
    if reference_coefficients is not None and np.iscomplexobj(reference_coefficients):raise ValueError('ROKS reference orbitals must be real')
    reference=c.copy() if reference_coefficients is None else np.array(reference_coefficients,dtype=float,copy=True)
    if restart is not None:
        with np.load(restart,allow_pickle=False) as data:
            metadata=json.loads(str(data['metadata']))
            if metadata['fingerprint']!=identity or not np.array_equal(data['occupations'],occ):raise ValueError('ROKS restart model mismatch')
            c=data['coefficients'].copy();reference=data['reference_coefficients'].copy()
            if ground_energy_hartree is None:ground_energy_hartree=metadata['ground_energy_hartree']
    s=mf.get_ovlp()
    for a in [c,reference]:
        if a.shape!=c.shape or not np.all(np.isfinite(a)) or not np.allclose(a.T@s@a,np.eye(len(a)),atol=1e-8,rtol=0):raise ValueError('ROKS orbitals are not orthonormal')
    if restart is not None and reference_coefficients is not None:
        if _singlet_overlap(np.asarray(reference_coefficients),reference,occ,s)<1-1e-8:
            raise ValueError('ROKS restart target reference mismatch')
    surface=OrbitalSurface(mf,c,occ,common=True)
    saved_certificate=metadata.get('solver_certificate') if restart is not None else None
    if initial_trust_radius is None and saved_certificate and saved_certificate.get('solver')==orbital_solver:
        if saved_certificate.get('status') in ('converged','iterating','maximum_iterations'):
            initial_trust_radius=saved_certificate.get('trust_radius')
            if initial_trust_radius is not None and initial_trust_radius<1e-6*maximum_step:initial_trust_radius=.5*maximum_step
        else:initial_trust_radius=.5*maximum_step
    def make_result(c,data,history,certificate):
        energy,g,*unused=data;residual=float(np.max(abs(g)));score=_singlet_overlap(reference,c,occ,s)
        return PySCFROKSResult(c.copy(),occ.copy(),reference.copy(),energy,ground_energy_hartree,residual,
            certificate['status']=='converged',len(history)-1,score,identity,history,s.copy(),solver_certificate=certificate)
    def save(c,data,history,certificate):
        if checkpoint is not None:make_result(c,data,history,certificate).save_checkpoint(checkpoint)
    initialization_history=()
    if initialize_closed_orbitals and restart is None:
        c,initialization_history=relax_closed_orbitals(surface,c,max_step=maximum_step,callback=save)
    c,data,history,certificate=solve(surface,c,lambda x:_singlet_overlap(reference,x,occ,s),
        tolerance=orbital_tolerance,max_iterations=maximum_iterations,max_step=maximum_step,
        min_reference_overlap=minimum_reference_overlap,min_step_overlap=minimum_step_overlap,
        step_overlap=lambda a,b:_singlet_overlap(a,b,occ,s),callback=save,
        initial_radius=initial_trust_radius,solver=orbital_solver)
    if certificate['status']!='converged' and freeze_and_release and orbital_solver=='trust':
        previous_history=history;previous_status=certificate['status']
        def phased_save(c,data,rows,details):
            combined=previous_history+tuple(dict(row,iteration=len(previous_history)+i) for i,row in enumerate(rows))
            save(c,data,combined,details)
        seed,frozen_history=relax_closed_orbitals(surface,reference.copy(),max_step=maximum_step,callback=phased_save)
        previous_history=history+tuple(dict(row,iteration=len(history)+i) for i,row in enumerate(frozen_history))
        c,data,recovered,certificate=solve(surface,seed,lambda x:_singlet_overlap(reference,x,occ,s),
            tolerance=orbital_tolerance,max_iterations=maximum_iterations,max_step=maximum_step,
            min_reference_overlap=minimum_reference_overlap,min_step_overlap=minimum_step_overlap,
            step_overlap=lambda a,b:_singlet_overlap(a,b,occ,s),callback=phased_save,
            initial_radius=.5*maximum_step,solver=orbital_solver)
        history=previous_history+tuple(dict(row,iteration=len(previous_history)+i,phase='released_orbitals') for i,row in enumerate(recovered))
        certificate['initialization_recovery']=dict(previous_status=previous_status,
            frozen_iterations=len(frozen_history)-1,final_space='All nonredundant rotations released')
    if certificate['status']=='converged' and (certify_index or target_saddle_index is not None):
        curvature=curvature_certificate(data[2],surface.dimension,index_threshold)
        certificate['curvature']=curvature
        if target_saddle_index is not None and curvature['index']!=target_saddle_index:
            certificate['status']='saddle_index_mismatch'
            certificate['requested_saddle_index']=int(target_saddle_index)
    if initialization_history:
        certificate['closed_orbital_preparation_iterations']=len(initialization_history)-1
        history=initialization_history+tuple(dict(row,iteration=len(initialization_history)+i) for i,row in enumerate(history))
    result=make_result(c,data,history,certificate)
    if checkpoint is not None:result.save_checkpoint(checkpoint)
    if strict and not result.converged:raise PySCFROKSConvergenceError(result)
    if result.converged and (include_gradient or include_hessian):
        from ._oo_nuclear import nuclear_response
        gradient,hessian,certificate=nuclear_response(surface,c,hessian=include_hessian)
        result=replace(result,gradient_hartree_per_bohr=gradient,hessian_hartree_per_bohr2=hessian,response_certificate=certificate)
    return result
