"""Variational ROKS surfaces with a frozen DFT-derived response and energy anchor.

E_state(R) = E_GFN2+frozen_response,ROKS(R) + offset_reference.
The offset matches a frozen-orbital or fully relaxed DFT excitation at the
anchor geometry, explicitly identified by ``kind``.
It is constant on this potential-energy surface. All orbital optimization,
forces and the analytic nuclear Hessian use the SAME frozen response functional.
This is not the geometry-by-geometry post-hoc frozen-DFT relaxation formula.
"""
from dataclasses import dataclass,replace,field
from hashlib import sha256
import numpy as np
from .spin_reference import FrozenSpinReference,validate_spin_reference


@dataclass(frozen=True,repr=False)
class FrozenROKSAnchor:
    response_reference: FrozenSpinReference
    occupations: np.ndarray
    coordinates_bohr: np.ndarray
    reference_excitation_hartree: float
    model_excitation_hartree: float
    provenance: str
    kind: str = field(default='frozen_orbitals',kw_only=True)

    def __repr__(self):return f'FrozenROKSAnchor({self.fingerprint})'

    def __post_init__(self):
        r=self.response_reference;o=np.asarray(self.occupations,dtype=float)
        x=np.asarray(self.coordinates_bohr,dtype=float)
        if (not isinstance(r,FrozenSpinReference) or o.shape!=(2,len(r.orbital_shell))
                or not np.all((o==0)|(o==1))
                or np.count_nonzero(o[0]-o[1]==1)!=1
                or np.count_nonzero(o[1]-o[0]==1)!=1
                or x.shape!=(len(r.atomic_numbers),3) or not np.all(np.isfinite(x))
                or not np.isfinite(self.reference_excitation_hartree)
                or not np.isfinite(self.model_excitation_hartree)
                or not isinstance(self.provenance,str) or not self.provenance.strip()
                or self.kind not in {'frozen_orbitals','relaxed_vertical'}):
            raise ValueError('invalid frozen ROKS anchor')
        for name,a in [('occupations',o),('coordinates_bohr',x)]:
            a=a.copy();a.setflags(write=False);object.__setattr__(self,name,a)

    @property
    def offset_hartree(self):
        return self.reference_excitation_hartree-self.model_excitation_hartree

    @property
    def fingerprint(self):
        h=sha256(self.response_reference.fingerprint.encode())
        for a in (self.occupations,self.coordinates_bohr):h.update(np.ascontiguousarray(a).tobytes())
        for v in (self.reference_excitation_hartree,self.model_excitation_hartree):h.update(float(v).hex().encode())
        h.update(self.provenance.encode())
        if self.kind!='frozen_orbitals':h.update(self.kind.encode())
        return h.hexdigest()


@dataclass(frozen=True)
class AnchoredROKSResult:
    reference: object
    anchor: FrozenROKSAnchor

    def __post_init__(self):
        from .roks import ROKSSingletResult
        if not isinstance(self.reference,ROKSSingletResult):raise TypeError('anchored ROKS needs a common-orbital singlet result')
        validate_spin_reference(self.reference,self.anchor.response_reference)
        if not np.array_equal(self.reference.occupations,self.anchor.occupations):
            raise ValueError('ROKS anchor occupations differ from the selected state')

    def __getattr__(self,name):return getattr(self.reference,name)

    @property
    def energy_hartree(self):return self.reference.energy_hartree+self.anchor.offset_hartree

    @property
    def excitation_energy_hartree(self):
        value=self.reference.excitation_energy_hartree
        return None if value is None else value+self.anchor.offset_hartree

    def to_apoc(self,state_id=1):
        state=self.reference.to_apoc(state_id);w=state.mixed.wavefunction
        w=replace(w,total_energy_hartree=self.energy_hartree,model=w.model+'.anchor.'+self.anchor.fingerprint)
        return replace(state,mixed=replace(state.mixed,wavefunction=w),energy_hartree=self.energy_hartree)


def evaluate_anchored_roks(calculator,coordinates_angstrom,anchor,*,reference_result=None,**options):
    from .roks import evaluate_gfn2_roks_singlet
    if not isinstance(anchor,FrozenROKSAnchor):raise TypeError('anchor must be a FrozenROKSAnchor')
    if reference_result is not None:
        if not isinstance(reference_result,AnchoredROKSResult) or reference_result.anchor.fingerprint!=anchor.fingerprint:
            raise ValueError('retained state belongs to a different anchored potential')
        reference_result=reference_result.reference
    result=evaluate_gfn2_roks_singlet(calculator,coordinates_angstrom,*anchor.occupations,
        spin_reference=anchor.response_reference,reference_result=reference_result,**options)
    return AnchoredROKSResult(result,anchor)


def evaluate_anchored_roks_analytic_hessian(calculator,coordinates_angstrom,anchor,*,reference_result=None,**options):
    from .deltascf_analytic_hessian import evaluate_gfn2_deltascf_analytic_hessian
    if not isinstance(anchor,FrozenROKSAnchor):raise TypeError('anchor must be a FrozenROKSAnchor')
    if reference_result is None:
        reference_result=evaluate_anchored_roks(calculator,coordinates_angstrom,anchor,gradient=True,**options)
    if not isinstance(reference_result,AnchoredROKSResult) or reference_result.anchor.fingerprint!=anchor.fingerprint:
        raise ValueError('analytic Hessian requested for a different anchored potential')
    h=evaluate_gfn2_deltascf_analytic_hessian(calculator,coordinates_angstrom,*anchor.occupations,
        mode='roks',reference_result=reference_result.reference,spin_reference=anchor.response_reference)
    return replace(h,reference=reference_result,method='analytic_frozen_reference_roks_schur_complement')


def _request(anchor,request_hash):return sha256((request_hash+':'+anchor.fingerprint).encode()).hexdigest()


def save_anchored_roks_point(path,result,*,request_hash,coordinates_angstrom):
    from .deltascf_derivatives import save_derivative_point
    if not isinstance(result,AnchoredROKSResult):raise TypeError('expected an anchored ROKS result')
    save_derivative_point(path,result.reference,request_hash=_request(result.anchor,request_hash),
                          coordinates_angstrom=coordinates_angstrom)


def load_anchored_roks_point(path,anchor,*,request_hash,coordinates_angstrom,require_gradient=True):
    from .deltascf_derivatives import load_derivative_point
    r=load_derivative_point(path,request_hash=_request(anchor,request_hash),
        coordinates_angstrom=coordinates_angstrom,require_gradient=require_gradient)
    return AnchoredROKSResult(r,anchor)


def save_frozen_roks_anchor(path,anchor):
    """Persist the complete frozen potential, without executable pickle data."""
    from pathlib import Path
    from .spin_reference import FrozenResponseReference,FrozenTensorResponseReference,FrozenFockResponseReference
    r=anchor.response_reference
    if isinstance(r,(FrozenTensorResponseReference,FrozenFockResponseReference)):
        raise ValueError('persist extended response models with save_response_reference; energy anchoring is not supported')
    data=dict(schema='mouse.frozen_roks_anchor.v1',fingerprint=anchor.fingerprint,
        atomic_numbers=r.atomic_numbers,orbital_shell=r.orbital_shell,coupling_hartree=r.coupling_hartree,
        source_model=r.source_model,response_exchange_scale=r.response_exchange_scale,
        occupations=anchor.occupations,coordinates_bohr=anchor.coordinates_bohr,
        reference_excitation_hartree=anchor.reference_excitation_hartree,
        model_excitation_hartree=anchor.model_excitation_hartree,provenance=anchor.provenance,
        kind=anchor.kind,
        response_kind='charge_spin' if isinstance(r,FrozenResponseReference) else 'spin')
    if isinstance(r,FrozenResponseReference):
        data.update(charge_correction_hartree=r.charge_correction_hartree,
                    reference_shell_populations=r.reference_shell_populations)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as stream:np.savez_compressed(stream,**data)
    temporary.replace(path)


def load_frozen_roks_anchor(path):
    from .spin_reference import FrozenResponseReference
    with np.load(path,allow_pickle=False) as data:
        if str(data['schema'])!='mouse.frozen_roks_anchor.v1':raise ValueError('unknown frozen ROKS anchor format')
        common=dict(atomic_numbers=data['atomic_numbers'],orbital_shell=data['orbital_shell'],
            coupling_hartree=data['coupling_hartree'],source_model=str(data['source_model']),
            response_exchange_scale=float(data['response_exchange_scale']))
        kind=str(data['response_kind'])
        if kind=='charge_spin':r=FrozenResponseReference(**common,charge_correction_hartree=data['charge_correction_hartree'],reference_shell_populations=data['reference_shell_populations'])
        elif kind=='spin':r=FrozenSpinReference(**common)
        else:raise ValueError('unknown frozen response kind')
        anchor=FrozenROKSAnchor(r,data['occupations'],data['coordinates_bohr'],
            float(data['reference_excitation_hartree'] if 'reference_excitation_hartree' in data else data['dft_frozen_excitation_hartree']),
            float(data['model_excitation_hartree'] if 'model_excitation_hartree' in data else data['model_frozen_excitation_hartree']),str(data['provenance']),
            kind=str(data['kind']) if 'kind' in data else 'frozen_orbitals')
        if anchor.fingerprint!=str(data['fingerprint']):raise ValueError('frozen ROKS anchor checksum mismatch')
    return anchor
