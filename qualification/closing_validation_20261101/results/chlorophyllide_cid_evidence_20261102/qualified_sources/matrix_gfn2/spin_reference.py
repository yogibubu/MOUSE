"""Frozen molecular shell-spin kernels for variational ROKS experiments.

The coupling is a parameter of ONE potential-energy surface. It is held fixed
under nuclear displacement; the Mulliken metric, orbital response and every
other GFN2 term still vary analytically. Rebuilding it at each geometry defines
a different surface whose kernel derivatives are not supplied here.
"""
from dataclasses import dataclass
from hashlib import sha256
import numpy as np
from .spin_polarization import ShellSpinPolarization


@dataclass(frozen=True,repr=False)
class FrozenSpinReference:
    atomic_numbers: np.ndarray
    orbital_shell: np.ndarray
    coupling_hartree: np.ndarray
    source_model: str
    response_exchange_scale: float = .9

    def __post_init__(self):
        z=np.asarray(self.atomic_numbers);shell=np.asarray(self.orbital_shell)
        if (z.ndim!=1 or not len(z) or z.dtype.kind not in 'iu' or np.any(z<=0)
                or shell.ndim!=1 or not len(shell) or shell.dtype.kind not in 'iu'
                or not isinstance(self.source_model,str) or not self.source_model.strip()
                or isinstance(self.response_exchange_scale,(bool,np.bool_))
                or not np.isfinite(self.response_exchange_scale)
                or not 0<=self.response_exchange_scale<=1):
            raise ValueError('invalid frozen shell-spin reference identity')
        w=np.asarray(self.coupling_hartree,dtype=float)
        ShellSpinPolarization(np.eye(len(shell)),shell,w)
        if not np.array_equal(np.unique(shell),np.arange(len(w))):
            raise ValueError('frozen spin reference must cover every shell')
        for name in ('atomic_numbers','orbital_shell','coupling_hartree'):
            a=np.array(getattr(self,name),copy=True);a.setflags(write=False)
            object.__setattr__(self,name,a)

    @property
    def fingerprint(self):
        h=sha256()
        for a in (self.atomic_numbers,self.orbital_shell,self.coupling_hartree):
            v=np.ascontiguousarray(a,dtype=np.float64)
            h.update(str(v.shape).encode());h.update(v.tobytes())
        h.update(self.source_model.encode());h.update(float(self.response_exchange_scale).hex().encode())
        return h.hexdigest()

    def __repr__(self):
        return f'FrozenSpinReference({self.fingerprint})'

    def bind(self,state,partition):
        if (not np.array_equal(state.atomic_numbers,self.atomic_numbers)
                or not np.array_equal(partition.orbital_shell,self.orbital_shell)):
            raise ValueError('frozen spin reference does not match atoms and shell order')
        return ShellSpinPolarization(state.overlap,self.orbital_shell,self.coupling_hartree)


def validate_spin_reference(result,reference):
    expected=None if reference is None else reference.fingerprint
    if getattr(result,'spin_reference_fingerprint',None)!=expected:
        raise ValueError('retained state and requested spin functional differ')


@dataclass(frozen=True,repr=False)
class FrozenResponseReference(FrozenSpinReference):
    """Frozen charge response correction in addition to the spin kernel.

    E_charge = .5 (n(P,S)-n_anchor).T U (n(P,S)-n_anchor).
    U and n_anchor remain fixed parameters; n(P,S) is the actual Mulliken
    shell population at every geometry and trial density.
    """
    charge_correction_hartree: np.ndarray | None = None
    reference_shell_populations: np.ndarray | None = None

    def __post_init__(self):
        super().__post_init__()
        u=np.asarray(self.charge_correction_hartree,dtype=float)
        n=np.asarray(self.reference_shell_populations,dtype=float)
        count=len(self.coupling_hartree)
        if (u.shape!=(count,count) or n.shape!=(count,)
                or not np.all(np.isfinite(u)) or not np.all(np.isfinite(n))
                or not np.allclose(u,u.T,atol=1e-12,rtol=0)):
            raise ValueError('invalid frozen charge-response correction')
        for name,a in [('charge_correction_hartree',u),('reference_shell_populations',n)]:
            a=a.copy();a.setflags(write=False);object.__setattr__(self,name,a)

    @property
    def fingerprint(self):
        h=sha256(super().fingerprint.encode())
        for a in (self.charge_correction_hartree,self.reference_shell_populations):
            h.update(np.ascontiguousarray(a,dtype=np.float64).tobytes())
        return h.hexdigest()

    def __repr__(self):return f'FrozenResponseReference({self.fingerprint})'

    def bind_charge(self,state,partition):
        self.bind(state,partition)  # also validate the chemical/shell identity
        return ShellChargeCorrection(state.overlap,self.orbital_shell,
                                     self.charge_correction_hartree,self.reference_shell_populations)


@dataclass(frozen=True)
class ShellChargeCorrection:
    overlap: np.ndarray
    orbital_shell: np.ndarray
    coupling_hartree: np.ndarray
    reference_populations: np.ndarray

    def moments(self,p,metric=None):
        s=self.overlap if metric is None else metric
        return np.bincount(self.orbital_shell,weights=np.sum(p*s,axis=1),
                           minlength=len(self.coupling_hartree))

    def potential(self,m,metric=None):
        s=self.overlap if metric is None else metric
        v=(self.coupling_hartree@m)[self.orbital_shell]
        return .5*s*(v[:,None]+v[None,:])

    def evaluate(self,total_density):
        dn=self.moments(total_density)-self.reference_populations
        return self.potential(dn),.5*float(dn@self.coupling_hartree@dn)

    def response(self,density_change):return self.potential(self.moments(density_change))

    def metric_potential(self,p):
        v=(self.coupling_hartree@(self.moments(p)-self.reference_populations))[self.orbital_shell]
        return .5*p*(v[:,None]+v[None,:])

    def geometry(self,total_density,ds,d2s):
        """Fixed-AO-density first Fock and second energy geometry derivatives."""
        m=self.moments(total_density)-self.reference_populations
        dm=np.array([self.moments(total_density,x) for x in ds])
        v=(self.coupling_hartree@m)[self.orbital_shell]
        dv=(dm@self.coupling_hartree.T)[:,self.orbital_shell]
        df=.5*(ds*(v[:,None]+v[None,:])+self.overlap[None]*(dv[:,:,None]+dv[:,None,:]))
        rr=dm@self.coupling_hartree@dm.T
        for a in range(len(ds)):
            for b in range(len(ds)):
                rr[a,b]+=float(m@self.coupling_hartree@self.moments(total_density,d2s[a,b]))
        return df,rr


@dataclass(frozen=True,repr=False)
class FrozenTensorResponseReference(FrozenResponseReference):
    """Shell charge/spin response plus invariant on-site spin anisotropy.

    Columns are sp-vector and traceless pp-tensor coefficients, per atom.
    Every component within one irreducible block shares its coefficient.
    The spin term vanishes for a closed shell. Charge anisotropy requires the
    ground density to relax under this same functional; there is no energy offset.
    """
    onsite_spin_hartree: np.ndarray | None = None
    onsite_charge_hartree: np.ndarray | None = None

    def __post_init__(self):
        super().__post_init__()
        a=np.asarray(self.onsite_spin_hartree,dtype=float)
        if a.shape!=(len(self.atomic_numbers),2) or not np.all(np.isfinite(a)):
            raise ValueError('invalid on-site tensor spin coefficients')
        a=a.copy();a.setflags(write=False);object.__setattr__(self,'onsite_spin_hartree',a)
        q=np.zeros_like(a) if self.onsite_charge_hartree is None else np.asarray(self.onsite_charge_hartree,dtype=float)
        if q.shape!=a.shape or not np.all(np.isfinite(q)):raise ValueError('invalid on-site tensor charge coefficients')
        q=q.copy();q.setflags(write=False);object.__setattr__(self,'onsite_charge_hartree',q)

    @property
    def fingerprint(self):
        return sha256(super().fingerprint.encode()+self.onsite_spin_hartree.tobytes()+self.onsite_charge_hartree.tobytes()).hexdigest()

    def bind(self,state,partition):
        from .tensor_spin import onsite_spin_generators,TensorSpinPolarization
        base=super().bind(state,partition);a,groups=onsite_spin_generators(state)
        return TensorSpinPolarization(base,a,self.onsite_spin_hartree[groups[:,0],groups[:,1]])

    def bind_charge(self,state,partition):
        from .tensor_spin import onsite_spin_generators,TensorSpinPolarization,TensorChargeCorrection
        base=ShellChargeCorrection(state.overlap,self.orbital_shell,self.charge_correction_hartree,self.reference_shell_populations)
        self.bind(state,partition)
        a,groups=onsite_spin_generators(state)
        # The tensor only needs the current metric, supplied through the validated shell object.
        from .spin_polarization import ShellSpinPolarization
        tensor=TensorSpinPolarization(ShellSpinPolarization(state.overlap,self.orbital_shell,self.coupling_hartree),a,self.onsite_charge_hartree[groups[:,0],groups[:,1]])
        return TensorChargeCorrection(base,tensor)


def save_response_reference(path,reference):
    """Save a complete no-offset response functional, without pickle."""
    from pathlib import Path
    from .nonlocal_response import FrozenNonlocalResponseReference,save_nonlocal_response_reference
    if isinstance(reference,FrozenNonlocalResponseReference):
        return save_nonlocal_response_reference(path,reference)
    if not isinstance(reference,FrozenResponseReference):raise TypeError('expected a frozen response reference')
    data=dict(schema='mouse.frozen_response.v1',atomic_numbers=reference.atomic_numbers,
        orbital_shell=reference.orbital_shell,coupling_hartree=reference.coupling_hartree,
        source_model=reference.source_model,response_exchange_scale=reference.response_exchange_scale,
        charge_correction_hartree=reference.charge_correction_hartree,
        reference_shell_populations=reference.reference_shell_populations,
        kind=('tensor' if isinstance(reference,FrozenTensorResponseReference) else ('fock' if isinstance(reference,FrozenFockResponseReference) else 'shell')),fingerprint=reference.fingerprint)
    if isinstance(reference,FrozenTensorResponseReference):
        data['onsite_spin_hartree']=reference.onsite_spin_hartree
        data['onsite_charge_hartree']=reference.onsite_charge_hartree
    if isinstance(reference,FrozenFockResponseReference):data['shell_potential_hartree']=reference.shell_potential_hartree
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as stream:np.savez_compressed(stream,**data)
    temporary.replace(path)


def load_response_reference(path):
    with np.load(path,allow_pickle=False) as d:
        if str(d['schema']) in {'mouse.nonlocal_response.v1','mouse.local_nonlocal_response.v1'}:
            from .nonlocal_response import load_nonlocal_response_reference
            return load_nonlocal_response_reference(path)
        if str(d['schema'])!='mouse.frozen_response.v1':raise ValueError('unknown response format')
        kind=str(d['kind'])
        if kind not in {'shell','tensor','fock'}:raise ValueError('unknown response kind')
        kwargs={k:d[k].copy() for k in ['atomic_numbers','orbital_shell','coupling_hartree','charge_correction_hartree','reference_shell_populations']}
        kwargs.update(source_model=str(d['source_model']),response_exchange_scale=float(d['response_exchange_scale']))
        if kind=='tensor':
            kwargs['onsite_spin_hartree']=d['onsite_spin_hartree'].copy()
            kwargs['onsite_charge_hartree']=d['onsite_charge_hartree'].copy()
        if kind=='fock':kwargs['shell_potential_hartree']=d['shell_potential_hartree'].copy()
        cls={'tensor':FrozenTensorResponseReference,'fock':FrozenFockResponseReference,'shell':FrozenResponseReference}[kind]
        r=cls(**kwargs)
        if r.fingerprint!=str(d['fingerprint']):raise ValueError('response checksum mismatch')
        return r


@dataclass(frozen=True)
class LinearShellChargeCorrection:
    base: ShellChargeCorrection
    shell_potential_hartree: np.ndarray

    def evaluate(self,p):
        f,e=self.base.evaluate(p);n=self.base.moments(p)
        v=self.shell_potential_hartree[self.base.orbital_shell]
        return f+.5*self.base.overlap*(v[:,None]+v[None,:]),e+float(self.shell_potential_hartree@n)

    def response(self,dp):return self.base.response(dp)

    def metric_potential(self,p):
        v=self.shell_potential_hartree[self.base.orbital_shell]
        return self.base.metric_potential(p)+.5*p*(v[:,None]+v[None,:])

    def geometry(self,p,ds,d2s):
        f,rr=self.base.geometry(p,ds,d2s);v=self.shell_potential_hartree[self.base.orbital_shell]
        f=f+.5*ds*(v[:,None]+v[None,:])
        rr=rr+np.einsum('ij,abij->ab',.5*p*(v[:,None]+v[None,:]),d2s)
        return f,rr


@dataclass(frozen=True,repr=False)
class FrozenFockResponseReference(FrozenResponseReference):
    """Fixed first- AND second-order corrections, with no excitation offset.

    The linear shell potential changes the Hamiltonian and therefore requires
    ground-state relaxation under the corrected functional, just like ROKS.
    """
    shell_potential_hartree: np.ndarray | None = None

    def __post_init__(self):
        super().__post_init__();v=np.asarray(self.shell_potential_hartree,dtype=float)
        if v.shape!=(len(self.coupling_hartree),) or not np.all(np.isfinite(v)):
            raise ValueError('invalid shell Fock potential')
        v=v.copy();v.setflags(write=False);object.__setattr__(self,'shell_potential_hartree',v)

    @property
    def fingerprint(self):return sha256(super().fingerprint.encode()+b'linear-shell-fock'+self.shell_potential_hartree.tobytes()).hexdigest()

    def bind_charge(self,state,partition):
        base=super().bind_charge(state,partition)
        return LinearShellChargeCorrection(base,self.shell_potential_hartree)
