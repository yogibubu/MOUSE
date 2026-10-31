"""Coherent real spin-singlet CSF expansions; no energy-based state matching."""
from dataclasses import dataclass
import numpy as np
from .deltascf import SpinAdaptedSinglet,singlet_csf_overlap
from .state import StateMatch,StateContinuityError

@dataclass(frozen=True)
class SingletCSFExpansion:
    state_id: int
    csfs: tuple[SpinAdaptedSinglet,...]
    coefficients: np.ndarray
    energy_hartree: float | None = None

    def __post_init__(self):
        if isinstance(self.state_id,bool) or int(self.state_id)!=self.state_id or self.state_id<1:
            raise ValueError('expansion state identifier must be a positive integer')
        csfs=tuple(self.csfs);z=np.asarray(self.coefficients)
        if not csfs or not all(isinstance(c,SpinAdaptedSinglet) for c in csfs) or np.iscomplexobj(z) or z.shape!=(len(csfs),) or not np.all(np.isfinite(z)):
            raise ValueError('invalid real spin-singlet expansion')
        metric=csfs[0].mixed.wavefunction.overlap_ao
        gram=np.array([[np.prod(singlet_csf_overlap(a,b,metric)) for b in csfs] for a in csfs])
        if not np.allclose(gram,np.eye(len(csfs)),atol=1e-7,rtol=0) or not np.isclose(z@z,1.,atol=1e-9,rtol=0):
            raise ValueError('expansion requires orthonormal CSFs and normalized coefficients')
        z=np.array(z,dtype=float,copy=True);z.setflags(write=False)
        object.__setattr__(self,'coefficients',z);object.__setattr__(self,'csfs',csfs)

    @property
    def spin_squared(self):return 0.


def multicsf_overlap(reference,candidate,overlap_reference_candidate):
    """Absolute coherent overlap and signed phase, including interference."""
    amplitude=sum(x*y*np.prod(singlet_csf_overlap(a,b,overlap_reference_candidate)) for x,a in zip(reference.coefficients,reference.csfs) for y,b in zip(candidate.coefficients,candidate.csfs))
    if abs(amplitude)>1+1e-5:raise ValueError('expansion overlap exceeds normalized bound')
    return float(min(1.,abs(amplitude))),1. if amplitude>=0 else -1.


def follow_multicsf_state(reference,candidates,overlap_reference_candidate,*,minimum_overlap=.9,ambiguity_margin=.05,strict=True):
    if not 0<=minimum_overlap<=1 or not 0<=ambiguity_margin<=1:raise ValueError('invalid expansion tracking thresholds')
    candidates=tuple(candidates)
    if not candidates:raise StateContinuityError('no spin-singlet expansion candidates supplied')
    if len({s.state_id for s in candidates})!=len(candidates):raise ValueError('duplicate expansion state identifiers')
    values=[multicsf_overlap(reference,c,overlap_reference_candidate) for c in candidates];order=np.argsort([-s[0] for s in values],kind='stable');i=order[0];runner=values[order[1]][0] if len(order)>1 else 0.
    score,phase=values[i];e0,e1=reference.energy_hartree,candidates[i].energy_hartree
    match=StateMatch(reference.state_id,candidates[i].state_id,score,phase,None if e0 is None or e1 is None else e1-e0,runner,score-runner,score>=minimum_overlap,score-runner<ambiguity_margin)
    if strict and (not match.continuous or match.ambiguous):raise StateContinuityError(f'CSF expansion continuity failed: overlap={score:.6f}, margin={score-runner:.6f}')
    return candidates[i],match
