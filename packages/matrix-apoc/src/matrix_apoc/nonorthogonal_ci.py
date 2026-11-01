"""Coherent state following for nonorthogonal, explicitly spin-pure singlets."""
from dataclasses import dataclass
import numpy as np
from .state import StateMatch,StateContinuityError


def _amplitude(left,right,cross):
    return float(sum(x*y*np.linalg.det(a.T@cross@c)*np.linalg.det(b.T@cross@d) for x,(a,b) in left for y,(c,d) in right))


@dataclass(frozen=True)
class NonOrthogonalSingletCIState:
    state_id: int
    determinants: tuple
    coefficients: np.ndarray
    overlap_ao: np.ndarray
    energy_hartree: float | None = None

    def __post_init__(self):
        if isinstance(self.state_id,(bool,np.bool_)) or int(self.state_id)!=self.state_id or self.state_id<1:raise ValueError('invalid CI state identifier')
        if np.iscomplexobj(self.overlap_ao):raise ValueError('complex CI AO metric is unsupported')
        z=np.asarray(self.coefficients);s=np.asarray(self.overlap_ao,dtype=float);determinants=tuple(self.determinants)
        if s.ndim!=2 or s.shape[0]!=s.shape[1] or not np.all(np.isfinite(s)) or not np.allclose(s,s.T,atol=1e-10,rtol=0):raise ValueError('invalid CI AO metric')
        if np.linalg.eigvalsh(s).min()<=0:raise ValueError('CI AO metric is not positive definite')
        if np.iscomplexobj(z) or z.shape!=(len(determinants),) or not len(z) or not np.all(np.isfinite(z)):raise ValueError('invalid real CI coefficients')
        copied=[];no=None
        for a,b in determinants:
            if np.iscomplexobj(a) or np.iscomplexobj(b):raise ValueError('complex CI determinant is unsupported')
            a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
            if a.ndim!=2 or b.shape!=a.shape or len(a)!=len(s) or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):raise ValueError('invalid equal-spin CI determinant')
            if no is None:no=a.shape[1]
            if a.shape[1]!=no:raise ValueError('inconsistent CI electron counts')
            for c in (a,b):
                if not np.allclose(c.T@s@c,np.eye(no),atol=1e-7,rtol=0):raise ValueError('CI determinant orbitals are not orthonormal')
            a=a.copy();b=b.copy();a.setflags(write=False);b.setflags(write=False);copied.append((a,b))
        terms=list(zip(z,copied));norm=_amplitude(terms,terms,s)
        if abs(norm-1)>1e-7:raise ValueError('CI wavefunction is not normalized')
        # For M_s=0, <S²> = ||S_+ Psi||². Remove wedge-zero occupied
        # components before assembling raised determinants; no FCI expansion
        # in the enormous full orbital space is needed.
        raised=[]
        for weight,(a,b) in terms:
            for j in range(no):
                v=b[:,j]-a@(a.T@s@b[:,j]);length=float(np.sqrt(max(0.,v@s@v)))
                if length<1e-10:continue
                raised.append((weight*(-1)**j*length,(np.column_stack((a,v/length)),np.delete(b,j,axis=1))))
        ss=_amplitude(raised,raised,s) if raised else 0.
        if ss < -1e-7 or ss>1e-7:raise ValueError(f'CI wavefunction is not a spin-pure singlet: S²={ss:.9g}')
        z=np.array(z,dtype=float,copy=True);z.setflags(write=False);s=s.copy();s.setflags(write=False)
        object.__setattr__(self,'coefficients',z);object.__setattr__(self,'overlap_ao',s);object.__setattr__(self,'determinants',tuple(copied))

    @property
    def spin_squared(self):return 0.


def nonorthogonal_ci_overlap(reference,candidate,overlap_reference_candidate):
    cross=np.asarray(overlap_reference_candidate,dtype=float)
    if cross.shape!=(len(reference.overlap_ao),len(candidate.overlap_ao)) or not np.all(np.isfinite(cross)):raise ValueError('invalid cross-geometry CI overlap')
    amplitude=_amplitude(zip(reference.coefficients,reference.determinants),list(zip(candidate.coefficients,candidate.determinants)),cross)
    if abs(amplitude)>1+1e-5:raise ValueError('normalized CI overlap exceeds unity')
    return min(1.,abs(amplitude)),1. if amplitude>=0 else -1.


def follow_nonorthogonal_ci_state(reference,candidates,overlap_reference_candidate,*,minimum_overlap=.9,ambiguity_margin=.05,strict=True):
    if not 0<=minimum_overlap<=1 or not 0<=ambiguity_margin<=1:raise ValueError('invalid CI tracking thresholds')
    candidates=tuple(candidates)
    if not candidates:raise StateContinuityError('no nonorthogonal CI candidates')
    if len({c.state_id for c in candidates})!=len(candidates):raise ValueError('duplicate CI candidate identifiers')
    values=[nonorthogonal_ci_overlap(reference,c,overlap_reference_candidate) for c in candidates];order=np.argsort([-v[0] for v in values],kind='stable');i=int(order[0]);runner=values[order[1]][0] if len(order)>1 else 0.
    score,phase=values[i];e0,e1=reference.energy_hartree,candidates[i].energy_hartree
    match=StateMatch(reference.state_id,candidates[i].state_id,score,phase,None if e0 is None or e1 is None else e1-e0,runner,score-runner,score>=minimum_overlap,score-runner<ambiguity_margin)
    if strict and (not match.continuous or match.ambiguous):raise StateContinuityError(f'CI state continuity failed: overlap={score:.6f}, margin={score-runner:.6f}')
    return candidates[i],match
