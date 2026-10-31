"""Frozen AO-pair response in a differentiable co-rotating molecular frame.

This defines a LOCAL potential around the calibration geometry. DFT parameters
are frozen, not recalculated along the path. Proper least-squares rotation of
labelled nuclei transports all AO operators. Its first and second derivatives
come from the polar stationarity equation, including planar molecules. Linear
or ambiguous frames are rejected. Currently only s/p shells are supported.
"""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import numpy as np
from .spin_reference import FrozenResponseReference


def rotation_jet(coordinates, anchor, *, second=True):
    x=np.asarray(coordinates,dtype=float);y=np.asarray(anchor,dtype=float)
    x=x-x.mean(axis=0);y=y-y.mean(axis=0)
    c=x.T@y;u,_,vt=np.linalg.svd(c)
    r=u@np.diag([1.,1.,np.linalg.det(u@vt)])@vt
    h=.5*(r.T@c+c.T@r);eig,v=np.linalg.eigh(h)
    gaps=eig[:,None]+eig[None,:];mask=~np.eye(3,dtype=bool)
    if np.min(gaps[mask])<=1e-9*max(1.,np.max(abs(eig))):
        raise ValueError('molecular transport frame is singular or ambiguous')
    def solve(rhs):
        z=v.T@rhs@v;out=np.zeros((3,3));out[mask]=z[mask]/gaps[mask]
        return v@out@v.T
    dc=np.zeros((x.size,3,3))
    for a in range(len(x)):
        for k in range(3):dc[3*a+k,k]=y[a]
    d=np.array([r.T@z for z in dc]);omega=np.array([solve(z-z.T) for z in d])
    dr=r@omega
    if not second:return r,dr,None
    dh=d-omega@h
    d2r=np.empty((x.size,x.size,3,3))
    for a in range(x.size):
        for b in range(x.size):
            rhs=-omega[b]@d[a]-d[a].T@omega[b]-dh[b]@omega[a]-omega[a]@dh[b]
            d2r[a,b]=r@(omega[b]@omega[a]+solve(rhs))
    return r,dr,.5*(d2r+d2r.swapaxes(0,1))


class TransportedOperators:
    def __init__(self,state,anchor,operators):
        self.state=state;self.anchor=anchor;self.original=operators
        self.groups=[]
        for atom in range(len(state.atomic_numbers)):
            for label in dict.fromkeys(state.ao_shell):
                ids=[i for i,(a,s) in enumerate(zip(state.ao_atom,state.ao_shell)) if a==atom and s==label]
                if not ids:continue
                if str(label).endswith('s') and len(ids)==1:continue
                if not str(label).endswith('p') or len(ids)!=3:
                    raise ValueError('AO-pair transport currently supports s/p shells only')
                self.groups.append(ids)
        r,_,_=rotation_jet(state.coordinates_bohr,anchor,second=False)
        self.u=self.lift(r,identity=True)
        self.a=self.u@operators@self.u.T
        self._first=None;self._second=None

    def lift(self,r,identity=False):
        n=self.original.shape[-1];u=np.eye(n) if identity else np.zeros((n,n))
        for ids in self.groups:u[np.ix_(ids,ids)]=r
        return u

    def first(self):
        if self._first is None:
            _,dr,_=rotation_jet(self.state.coordinates_bohr,self.anchor,second=False)
            du=np.array([self.lift(r) for r in dr])
            self._first=np.array([z@self.original@self.u.T+self.u@self.original@z.T for z in du])
        return self._first

    def second(self,i,j):
        if self._second is None:
            _,dr,d2r=rotation_jet(self.state.coordinates_bohr,self.anchor)
            self._second=(np.array([self.lift(r) for r in dr]),d2r)
        du,d2r=self._second;z=self.lift(d2r[i,j])
        return (z@self.original@self.u.T+self.u@self.original@z.T
                +du[i]@self.original@du[j].T+du[j]@self.original@du[i].T)


class PairTerm:
    def __init__(self,transport,k,q0=None,v=None):
        self.transport=transport;self.s=transport.state.overlap;self.a=transport.a;self.k=k
        self.q0=np.zeros(len(k)) if q0 is None else q0
        self.v=np.zeros(len(k)) if v is None else v
        self.b=self.metric(self.s,self.a)
    @staticmethod
    def metric(s,a):return .5*(s@a+a@s)
    def moments(self,p):return np.einsum('aij,ij->a',self.b,p)
    def field(self,p):return self.k@(self.moments(p)-self.q0)+self.v
    def evaluate(self,p):
        q=self.moments(p);dq=q-self.q0
        return np.einsum('a,aij->ij',self.k@dq+self.v,self.b),.5*float(dq@self.k@dq)+float(self.v@q)
    def response(self,p):return np.einsum('a,aij->ij',self.k@self.moments(p),self.b)
    def response_batch(self,p):
        """Apply the same linear operator to a batch using matrix products."""
        p=np.asarray(p);b=self.b.reshape(len(self.b),-1)
        moments=p.reshape(len(p),-1)@b.T
        return ((moments@self.k.T)@b).reshape(p.shape)
    def metric_potential(self,p):
        a=np.einsum('a,aij->ij',self.field(p),self.a)
        return .5*(p@a+a@p)
    def explicit_gradient(self,p):
        # Remaining coordinate dependence after dS is contracted by native integrals.
        da=self.transport.first();db=self.metric(self.s,da)
        return np.einsum('a,raij,ij->r',self.field(p),db,p).reshape(-1,3)
    def geometry(self,p,ds,d2s):
        da=self.transport.first();db=self.metric(ds[:,None],self.a)+self.metric(self.s,da)
        dq=np.einsum('raij,ij->ra',db,p);v=self.field(p)
        df=np.einsum('a,raij->rij',v,db)+np.einsum('ra,aij->rij',dq@self.k.T,self.b)
        rr=dq@self.k@dq.T
        for i in range(len(ds)):
            for j in range(len(ds)):
                d2b=(self.metric(d2s[i,j],self.a)+self.metric(ds[i],da[j])
                     +self.metric(ds[j],da[i])+self.metric(self.s,self.transport.second(i,j)))
                rr[i,j]+=np.einsum('a,aij,ij',v,d2b,p)
        return df,rr


class PairSpin:
    def __init__(self,base,term):self.base=base;self.term=term
    def __getattr__(self,name):return getattr(self.base,name)
    def evaluate(self,p):
        f,e,m=self.base.evaluate(p);v,de=self.term.evaluate(p[0]-p[1])
        return f+np.array([v,-v]),e+de,m
    def response_batch(self,p):
        base=(self.base.response_batch(p) if hasattr(self.base,'response_batch')
              else np.asarray([self.base.evaluate(x)[0] for x in p]))
        v=self.term.response_batch(p[:,0]-p[:,1])
        return base+np.stack((v,-v),axis=1)
    def tensor_metric_potential(self,p):return self.term.metric_potential(p[0]-p[1])
    def explicit_gradient(self,p):return self.term.explicit_gradient(p[0]-p[1])
    def tensor_geometry(self,p,ds,d2s):return self.term.geometry(p[0]-p[1],ds,d2s)


@dataclass(frozen=True,repr=False)
class FrozenNonlocalResponseReference(FrozenResponseReference):
    """Frozen full AO-pair charge/spin/one-body correction; no energy offset."""
    anchor_coordinates_bohr: np.ndarray = None
    operators: np.ndarray = None
    pair_charge_hartree: np.ndarray = None
    pair_spin_hartree: np.ndarray = None
    pair_reference_moments: np.ndarray = None
    pair_linear_hartree: np.ndarray = None

    def __post_init__(self):
        super().__post_init__();n=len(self.orbital_shell)
        m=np.asarray(self.operators).shape[0]
        shapes={'anchor_coordinates_bohr':(len(self.atomic_numbers),3),'operators':(m,n,n),
                'pair_charge_hartree':(m,m),'pair_spin_hartree':(m,m),
                'pair_reference_moments':(m,),'pair_linear_hartree':(m,)}
        if np.any(self.coupling_hartree) or np.any(self.charge_correction_hartree):
            raise ValueError('full AO-pair reference requires zero shell corrections')
        for name,shape in shapes.items():
            a=np.array(getattr(self,name),dtype=float,copy=True)
            if a.shape!=shape or not np.all(np.isfinite(a)):raise ValueError('invalid '+name)
            if name in {'operators','pair_charge_hartree','pair_spin_hartree'} and not np.allclose(a,a.swapaxes(-1,-2),atol=1e-10,rtol=0):
                raise ValueError('non-symmetric '+name)
            a.setflags(write=False);object.__setattr__(self,name,a)
        rotation_jet(self.anchor_coordinates_bohr,self.anchor_coordinates_bohr,second=False)
    @property
    def fingerprint(self):
        h=sha256((super().fingerprint+':corotating_ao_pair_v1').encode())
        for name in ('anchor_coordinates_bohr','operators','pair_charge_hartree','pair_spin_hartree','pair_reference_moments','pair_linear_hartree'):
            a=getattr(self,name);h.update(str(a.shape).encode());h.update(a.tobytes())
        return h.hexdigest()
    def __repr__(self):return f'FrozenNonlocalResponseReference({self.fingerprint})'
    def bind(self,state,partition):
        base=super().bind(state,partition)
        t=TransportedOperators(state,self.anchor_coordinates_bohr,self.operators)
        return PairSpin(base,PairTerm(t,self.pair_spin_hartree))
    def bind_charge(self,state,partition):
        self.bind(state,partition)
        t=TransportedOperators(state,self.anchor_coordinates_bohr,self.operators)
        return PairTerm(t,self.pair_charge_hartree,self.pair_reference_moments,self.pair_linear_hartree)
    @classmethod
    def from_vertical_model(cls,state,partition,data,source_model):
        ns=len(partition.shell_atom)
        return cls(state.atomic_numbers,partition.orbital_shell,np.zeros((ns,ns)),source_model,.9,
                   np.zeros((ns,ns)),np.zeros(ns),state.coordinates_bohr,data['operators'],
                   data['charge'],data['spin'],data['q0'],data['linear'])


def save_nonlocal_response_reference(path,reference):
    if not isinstance(reference,FrozenNonlocalResponseReference):raise TypeError('expected full AO-pair reference')
    data={name:getattr(reference,name) for name in reference.__dataclass_fields__}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    schema='mouse.local_nonlocal_response.v1' if hasattr(reference,'frame_clusters') else 'mouse.nonlocal_response.v1'
    with tmp.open('wb') as stream:np.savez_compressed(stream,**data,schema=schema,fingerprint=reference.fingerprint)
    tmp.replace(path)


def load_nonlocal_response_reference(path):
    with np.load(path,allow_pickle=False) as data:
        schema=str(data['schema']);cls=FrozenNonlocalResponseReference
        if schema=='mouse.local_nonlocal_response.v1':
            from .local_frame_response import FrozenLocalFrameResponseReference
            cls=FrozenLocalFrameResponseReference
        elif schema!='mouse.nonlocal_response.v1':raise ValueError('unknown full AO-pair reference schema')
        fields={name:data[name] for name in cls.__dataclass_fields__}
        fields['source_model']=str(fields['source_model']);fields['response_exchange_scale']=float(fields['response_exchange_scale'])
        r=cls(**fields)
        if r.fingerprint!=str(data['fingerprint']):raise ValueError('full AO-pair reference checksum mismatch')
        return r
