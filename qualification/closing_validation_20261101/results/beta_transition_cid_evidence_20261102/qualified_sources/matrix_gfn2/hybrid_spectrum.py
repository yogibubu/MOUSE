"""Experimental DFT-gap replacement with unchanged semiempirical orbitals.

Omega_hybrid = Omega_SE + scale * gap_reference - gap_SE. The same replacement can
be applied to the diagonal of a semiempirical TDA matrix. Orbital matching
uses the actual cross-AO metric, with independent occupied/virtual assignments.
Nuclear derivatives are returned only when derivatives of the correction are
explicitly supplied; an energy-only reference never inherits SE derivatives.
"""
from dataclasses import dataclass, replace
from hashlib import sha256
import numpy as np
from scipy.optimize import linear_sum_assignment


def _array(value, *, dtype=float):
    if np.iscomplexobj(value):
        raise ValueError('hybrid spectra currently require real arrays')
    a=np.array(value,dtype=dtype,copy=True)
    if not np.all(np.isfinite(a)):
        raise ValueError('hybrid spectrum arrays must be finite')
    a.setflags(write=False)
    return a


def spectrum_fingerprint(coefficients, overlap, occupations, energies):
    h=sha256()
    for a in (coefficients,overlap,occupations,energies):
        v=np.ascontiguousarray(a,dtype=np.float64)
        h.update(str(v.shape).encode());h.update(v.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class GapCorrectionJet:
    value_hartree: float
    source_fingerprint: str
    hole: int
    particle: int
    gradient_hartree_per_bohr: np.ndarray | None = None
    hessian_hartree_per_bohr2: np.ndarray | None = None

    def __post_init__(self):
        if (any(isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) or i<0
                for i in (self.hole,self.particle)) or self.hole==self.particle
                or not isinstance(self.source_fingerprint,str) or not self.source_fingerprint):
            raise ValueError('invalid gap-correction orbital identity')
        if not np.isfinite(self.value_hartree):raise ValueError('nonfinite gap correction')
        g=self.gradient_hartree_per_bohr;h=self.hessian_hartree_per_bohr2
        if g is not None:
            g=_array(g).reshape(-1);object.__setattr__(self,'gradient_hartree_per_bohr',g)
        if h is not None:
            h=_array(h)
            if (g is None or h.shape!=(len(g),len(g))
                    or not np.allclose(h,h.T,atol=1e-10,rtol=0)):
                raise ValueError('correction Hessian needs a matching gradient and symmetric dimensions')
            object.__setattr__(self,'hessian_hartree_per_bohr2',h)


@dataclass(frozen=True)
class OccupiedExtentScaling:
    """Experimental basis-compactness descriptor, without energy-target fitting.

    Radii are central spreads sqrt(<r^2>-|<r>|^2) of corresponding ground
    orbitals in bohr. Occupied reference levels scale relative to the lowest
    mapped virtual by (sigma_SE/sigma_DFT)**power; virtual spacings are retained.
    No clipping is used, so the radius-ratio law is smooth for positive radii.
    This is a model hypothesis, not a self-interaction correction derivation.
    """
    source_radii_bohr: np.ndarray
    reference_radii_bohr: np.ndarray
    power: float = 1.0

    def __post_init__(self):
        source=_array(self.source_radii_bohr);reference=_array(self.reference_radii_bohr)
        if (source.ndim!=1 or not len(source) or source.shape!=reference.shape
                or np.any(source<=0) or np.any(reference<=0)):
            raise ValueError('orbital extent scaling requires positive matched radii')
        if isinstance(self.power,(bool,np.bool_)) or not np.isfinite(self.power) or self.power<=0:
            raise ValueError('orbital extent power must be finite and positive')
        with np.errstate(over='ignore',invalid='ignore',under='ignore'):
            weights=(source/reference)**self.power
        if not np.all(np.isfinite(weights)) or np.any(weights<=0):
            raise ValueError('orbital extent power gives nonfinite or vanishing weights')
        object.__setattr__(self,'source_radii_bohr',source)
        object.__setattr__(self,'reference_radii_bohr',reference)

    @property
    def weights(self):
        return _array((self.source_radii_bohr/self.reference_radii_bohr)**self.power)


@dataclass(frozen=True)
class MatchedOrbitalSpectrum:
    source_energies_hartree: np.ndarray
    source_occupations: np.ndarray
    reference_energies_hartree: np.ndarray
    reference_indices: np.ndarray
    overlaps: np.ndarray
    margins: np.ndarray
    source_fingerprint: str
    reference_model: str
    minimum_overlap: float
    minimum_margin: float
    reference_energy_scale: float = 1.0
    extent_scaling: OccupiedExtentScaling | None = None

    def __post_init__(self):
        n=len(self.source_energies_hartree)
        indices=np.asarray(self.reference_indices)
        if (n<1 or indices.dtype.kind not in 'iu' or np.any(indices<0)
                or len(np.unique(indices))!=n or not self.source_fingerprint or not self.reference_model
                or not all(np.isfinite(x) and 0<=x<=1 for x in (self.minimum_overlap,self.minimum_margin))):
            raise ValueError('invalid matched spectrum identity or matching thresholds')
        if (isinstance(self.reference_energy_scale,(bool,np.bool_))
                or not np.isfinite(self.reference_energy_scale) or self.reference_energy_scale<=0):
            raise ValueError('reference orbital-energy scale must be finite and positive')
        for key in ('source_energies_hartree','source_occupations','reference_energies_hartree',
                    'reference_indices','overlaps','margins'):
            a=_array(getattr(self,key),dtype=int if key=='reference_indices' else float)
            if a.shape!=(n,):raise ValueError('inconsistent matched spectrum dimensions')
            object.__setattr__(self,key,a)
        if (not np.all((self.source_occupations==0)|(self.source_occupations==2))
                or np.any(self.overlaps<0) or np.any(self.overlaps>1+1e-5)
                or np.any(np.abs(self.margins)>1+1e-5)):
            raise ValueError('invalid matched orbital occupations or overlap scores')
        if self.extent_scaling is not None:
            if (not isinstance(self.extent_scaling,OccupiedExtentScaling)
                    or self.extent_scaling.source_radii_bohr.shape!=(n,)
                    or not np.any(self.source_occupations==0)):
                raise ValueError('extent descriptor does not match the orbital spectrum')
            # The energy anchor must itself have a reliable cross-basis mapping.
            virtual=np.flatnonzero(self.source_occupations==0)
            levels=np.sort(self.reference_energies_hartree[virtual])
            if len(levels)>1 and levels[1]-levels[0]<1e-8:
                raise ValueError('extent scaling requires an isolated reference virtual anchor')
            anchor=virtual[np.argmin(self.reference_energies_hartree[virtual])]
            self.validate_orbitals(np.array([anchor]))

    @property
    def effective_reference_energies_hartree(self):
        """Extent-transformed spectrum, before the common global lambda.

        Referencing all occupied shifts to one virtual energy guarantees that
        a common energy-origin change cancels from every configuration gap.
        """
        raw=self.reference_energies_hartree
        if self.extent_scaling is None:return raw
        occupied=self.source_occupations==2
        anchor=float(np.min(raw[~occupied]));energies=np.array(raw,copy=True)
        weights=self.extent_scaling.weights
        energies[occupied]=anchor+weights[occupied]*(raw[occupied]-anchor)
        return _array(energies)

    @property
    def qualified(self):
        return (self.overlaps>=self.minimum_overlap)&(self.margins>=self.minimum_margin)

    def validate_orbitals(self, indices):
        ids=np.asarray(indices)
        if ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=len(self.qualified)):
            raise ValueError('invalid source orbital indices')
        if not np.all(self.qualified[ids]):
            bad=ids[~self.qualified[ids]].tolist()
            raise ValueError(f'unqualified or ambiguous reference orbital mapping: {bad}')

    def configuration_corrections(self, occupied, virtual, *, require_qualified=True):
        i=np.asarray(occupied);a=np.asarray(virtual)
        for ids in (i,a):
            if ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=len(self.qualified)):
                raise ValueError('invalid configuration orbital indices')
        if i.shape!=a.shape or np.any(self.source_occupations[i]!=2) or np.any(self.source_occupations[a]!=0):
            raise ValueError('configuration must be an occupied to virtual promotion')
        if require_qualified:self.validate_orbitals(np.concatenate((i.ravel(),a.ravel())))
        # Scale the reference gap once, leaving the original SE contribution
        # subtracted in its own units. Do not scale the kernel or SE relaxation.
        effective=self.effective_reference_energies_hartree
        return (self.reference_energy_scale*(effective[a]-effective[i])
                -(self.source_energies_hartree[a]-self.source_energies_hartree[i]))

    def gap_correction(self,hole,particle,*,gradient=None,hessian=None):
        """Derivatives, when supplied, must be of scale*gap_ref - gap_SE.

        The global scale is constant. An enabled extent descriptor changes with
        geometry: its radii and anchor derivatives must also be included.
        Passing derivatives of a raw gap would describe another surface.
        """
        value=self.configuration_corrections(np.array([hole]),np.array([particle]))[0]
        return GapCorrectionJet(float(value),self.source_fingerprint,int(hole),int(particle),gradient,hessian)


def with_occupied_extent_scaling(mapping,source_radii_bohr,reference_radii_bohr,*,power=1.):
    """Attach the explicit extent hypothesis once; no automatic fitting.

    Radii must follow the source orbital order; reference radii must already
    be matched, using mapping.reference_indices. No state or target energy is
    accepted. Both TDA and ROKS consume the resulting same reference spectrum.
    """
    if not isinstance(mapping,MatchedOrbitalSpectrum) or mapping.extent_scaling is not None:
        raise ValueError('extent scaling requires a matched, unmodified reference spectrum')
    return replace(mapping,extent_scaling=OccupiedExtentScaling(source_radii_bohr,reference_radii_bohr,power))


def match_orbital_spectrum(source_coefficients,source_overlap,source_occupations,source_energies,
    reference_coefficients,reference_overlap,reference_occupations,reference_energies,
    cross_overlap,*,reference_model,minimum_overlap=.7,minimum_margin=.05,
    reference_energy_scale=1.0):
    """One-to-one matching within each occupation sector, independent of energy.

    cross_overlap has shape (source_nao, reference_nao). Full-electron reference
    cores and extra virtuals are allowed. Coordinates/atoms must be checked by
    the caller, since a bare AO overlap matrix has no geometry metadata.
    reference_energy_scale applies identically to TDA and ROKS reference gaps;
    it is independent of any scaling of exchange in a TDA response kernel.
    """
    cs,ss,os,es,cr,sr,orr,er,cross=map(_array,(source_coefficients,source_overlap,
        source_occupations,source_energies,reference_coefficients,reference_overlap,
        reference_occupations,reference_energies,cross_overlap))
    if (cs.ndim!=2 or cr.ndim!=2 or ss.shape!=(len(cs),len(cs))
            or sr.shape!=(len(cr),len(cr)) or cross.shape!=(len(cs),len(cr))
            or os.shape!=(cs.shape[1],) or es.shape!=os.shape
            or orr.shape!=(cr.shape[1],) or er.shape!=orr.shape
            or not all(np.all((o==0)|(o==2)) for o in (os,orr))
            or not all(np.any(o==0) and np.any(o==2) for o in (os,orr))
            or not all(np.allclose(s,s.T,atol=1e-10,rtol=0) for s in (ss,sr))
            or not np.allclose(cs.T@ss@cs,np.eye(cs.shape[1]),atol=2e-7,rtol=0)
            or not np.allclose(cr.T@sr@cr,np.eye(cr.shape[1]),atol=2e-7,rtol=0)):
        raise ValueError('invalid canonical spectra or cross-AO dimensions')
    if (not reference_model or not np.isfinite(minimum_overlap) or not 0<=minimum_overlap<=1
            or not np.isfinite(minimum_margin) or not 0<=minimum_margin<=1):
        raise ValueError('invalid reference model or matching thresholds')
    overlap=cs.T@cross@cr
    if np.linalg.svd(overlap,compute_uv=False)[0]>1+1e-5:
        raise ValueError('cross-AO metric is inconsistent with normalized orbitals')
    indices=np.empty(len(es),int);scores=np.empty(len(es));margins=np.empty(len(es))
    for occupation in (2,0):
        source=np.flatnonzero(os==occupation);reference=np.flatnonzero(orr==occupation)
        if len(reference)<len(source):raise ValueError('reference occupation sector is too small')
        weights=np.abs(overlap[np.ix_(source,reference)])
        rows,columns=linear_sum_assignment(-weights**2)
        for row,column in zip(rows,columns):
            idx=source[row];indices[idx]=reference[column];scores[idx]=weights[row,column]
            other=np.delete(weights[row],column)
            margins[idx]=scores[idx]-float(np.max(other,initial=0.))
    return MatchedOrbitalSpectrum(es,os,er[indices],indices,scores,margins,
        spectrum_fingerprint(cs,ss,os,es),str(reference_model),float(minimum_overlap),float(minimum_margin),
        reference_energy_scale)


def replace_tda_reference_gaps(matrix_hartree,occupied,virtual,mapping,*,require_qualified=True):
    a=_array(matrix_hartree)
    correction=mapping.configuration_corrections(occupied,virtual,require_qualified=require_qualified)
    if (correction.ndim!=1 or a.shape!=(len(correction),len(correction))
            or not np.allclose(a,a.T,atol=1e-10,rtol=0)):
        raise ValueError('invalid semiempirical TDA matrix')
    return a+np.diag(correction)


@dataclass(frozen=True)
class HybridExcitationEnergy:
    excitation_energy_hartree: float
    semiempirical_excitation_energy_hartree: float
    gap_correction_hartree: float
    gradient_hartree_per_bohr: np.ndarray | None = None
    hessian_hartree_per_bohr2: np.ndarray | None = None


def compose_hybrid_excitation(energy_hartree,correction,*,source_gradient=None,
    source_hessian=None,gradient=False,hessian=False):
    if not np.isfinite(energy_hartree):raise ValueError('nonfinite source excitation energy')
    g=h=None
    if gradient or hessian:
        if source_gradient is None or correction.gradient_hartree_per_bohr is None:
            raise ValueError('hybrid gradients require source AND gap-correction derivatives')
        g0=_array(source_gradient).reshape(-1);dg=correction.gradient_hartree_per_bohr
        if dg.shape!=g0.shape:raise ValueError('inconsistent gradient dimensions')
        g=_array(g0+dg)
    if hessian:
        if source_hessian is None or correction.hessian_hartree_per_bohr2 is None:
            raise ValueError('analytic hybrid Hessian requires the analytic gap-correction Hessian')
        h0=_array(source_hessian)
        if h0.shape!=(len(g),len(g)) or not np.allclose(h0,h0.T,atol=1e-10,rtol=0):
            raise ValueError('invalid source Hessian')
        h=_array(h0+correction.hessian_hartree_per_bohr2)
    return HybridExcitationEnergy(float(energy_hartree+correction.value_hartree),
        float(energy_hartree),correction.value_hartree,g,h)


def correct_roks_excitation(result,mapping,hole,particle,*,correction=None,
    ground_gradient=None,excitation_hessian=None,gradient=False,hessian=False):
    """Correct a retained stationary ROKS excitation, preserving its orbitals.

    Derivatives are of the GAP: supply the ground gradient separately, and
    supply an excited-minus-ground Hessian rather than a total-state Hessian.
    """
    from .roks import ROKSSingletResult
    if not isinstance(result,ROKSSingletResult) or not result.converged or result.excitation_energy_hartree is None:
        raise ValueError('a stationary retained ROKS state with a ground energy is required')
    s=result.reference_state
    if s is None or spectrum_fingerprint(s.coefficients,s.overlap,
        s.alpha_occupations+s.beta_occupations,s.orbital_energies_hartree)!=mapping.source_fingerprint:
        raise ValueError('ROKS and reference spectrum use different ground-state orbitals')
    expected=mapping.gap_correction(hole,particle)
    if (result.occupations[1,hole]-result.occupations[0,hole]!=1
            or result.occupations[0,particle]-result.occupations[1,particle]!=1):
        raise ValueError('gap correction promotion differs from the ROKS open orbitals')
    jet=expected if correction is None else correction
    if (jet.source_fingerprint!=expected.source_fingerprint or jet.hole!=hole or jet.particle!=particle
            or not np.isclose(jet.value_hartree,expected.value_hartree,atol=1e-12,rtol=0)):
        raise ValueError('gap correction does not match the mapped reference')
    g=None
    if gradient or hessian:
        if ground_gradient is None or result.gradient_hartree_per_bohr is None:
            raise ValueError('excitation gradients require excited AND ground gradients')
        excited=np.asarray(result.gradient_hartree_per_bohr).reshape(-1)
        ground=np.asarray(ground_gradient).reshape(-1)
        if excited.shape!=ground.shape:raise ValueError('inconsistent excited/ground gradient dimensions')
        g=excited-ground
    return compose_hybrid_excitation(result.excitation_energy_hartree,jet,
        source_gradient=g,source_hessian=excitation_hessian,gradient=gradient,hessian=hessian)
