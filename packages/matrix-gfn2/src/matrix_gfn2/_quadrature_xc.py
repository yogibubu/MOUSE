"""Exact frozen quadrature correction to an auxiliary XC preconditioner.

NumPy/SciPy only. No XC library or DFT backend is imported at evaluation.
The present adapter certifies vertical energies and orbital stationarity;
nuclear derivative evaluation is explicitly rejected until its jets are bound.
"""
from pathlib import Path
from hashlib import sha256
import json
import numpy as np
from scipy.linalg import eigh


class FrozenQuadratureXC:
    vertical_only=True

    def __init__(self,base,manifest,charge_kernel,spin_kernel,reference):
        self.base=base;self.reference=reference;self.manifest=Path(manifest)
        data=json.loads(self.manifest.read_text());self.n=len(base.state.overlap)
        if not np.allclose(data['coordinates_bohr'],base.state.coordinates_bohr,atol=1e-12,rtol=0):
            raise ValueError('quadrature correction is restricted to its anchor')
        self.i,self.j=np.tril_indices(self.n);self.w=np.where(self.i==self.j,1.,2**.5)
        self.ri=(np.load(charge_kernel,mmap_mode='r'),np.load(spin_kernel,mmap_mode='r'))
        self.ground=self.unpack(reference.pair_reference_moments)
        self.chunks=[]
        for item in data['chunks']:
            path=(self.manifest.parent/item['file']).resolve()
            if sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('frozen XC quadrature checksum mismatch')
            with np.load(path,allow_pickle=False) as z:self.chunks.append(tuple(z[k].copy() for k in ['phi','weight','charge','spin']))
        self.cache={};self.energy_cache={};self.span=None;self.polarization=_Spin(self)

    def __getattr__(self,key):return getattr(self.base,key)
    def pack(self,matrix):return matrix[self.i,self.j]*self.w
    def unpack(self,vector):
        a=np.zeros((self.n,self.n));a[self.i,self.j]=vector/self.w;a[self.j,self.i]=vector/self.w;return a
    @staticmethod
    def key(density):return sha256(np.ascontiguousarray(density).tobytes()).digest()
    @staticmethod
    def rho(phi,density):
        values,vectors=eigh(.5*(density+density.T));keep=abs(values)>1e-13
        if not np.any(keep):return np.zeros((4,phi.shape[1]))
        values=values[keep];vectors=vectors[:,keep];p=phi@vectors;rho=np.empty((4,phi.shape[1]));rho[0]=np.sum(p[0]**2*values[None,:],axis=1)
        for k in range(1,4):rho[k]=2*np.sum(p[k]*p[0]*values[None,:],axis=1)
        return rho
    def apply(self,density,channel,*,potential=True):
        key=(self.key(density),int(channel));previous=self.cache.get(key)
        if previous is not None and (not potential or previous[0] is not None):return previous
        if np.max(abs(density))<1e-15:return np.zeros_like(density),0.
        q=self.pack(density)
        if not potential and self.span is not None:
            vectors,pinverse,matrices=self.span;coeff=pinverse@q
            if np.linalg.norm(vectors@coeff-q)<1e-10:
                return None,.5*float(coeff@matrices[channel]@coeff)
        v=np.zeros_like(density) if potential else None;energy=0.
        for phi,weight,charge,spin in self.chunks:
            rho=self.rho(phi,density);fx=charge if channel==0 else spin;field=np.einsum('ijg,jg->ig',fx,rho,optimize=True);energy+=.5*float(np.sum(weight*np.sum(rho*field,axis=0)))
            if potential:
                v+=phi[0].T@((weight*field[0])[:,None]*phi[0])
                for k in range(1,4):
                    part=phi[k].T@((weight*field[k])[:,None]*phi[0]);v+=part+part.T
        approximate=self.ri[channel]@q;energy-=.5*float(q@approximate)
        if potential:v=.5*(v+v.T)-self.unpack(approximate)
        self.cache[key]=(v,float(energy));return v,float(energy)
    def fock_and_energy(self,spin_densities):
        p=np.asarray(spin_densities);f,e=self.base.fock_and_energy(p);vc,ec=self.apply(p.sum(axis=0)-self.ground,0);vs,es=self.apply(p[0]-p[1],1);energy=e+ec+es
        self.energy_cache[self.key(p)]=float(energy)
        return f+vc[None]+np.array([vs,-vs]),float(energy)
    def energy(self,spin_densities):
        p=np.asarray(spin_densities);key=self.key(p)
        if key not in self.energy_cache:
            e=self.base.fock_and_energy(p)[1];ec=self.apply(p.sum(axis=0)-self.ground,0,potential=False)[1];es=self.apply(p[0]-p[1],1,potential=False)[1];self.energy_cache[key]=float(e+ec+es)
        return self.energy_cache[key]
    def density_response_kernel(self,state):
        kernel=self.base.density_response_kernel(state)
        def action(dp):return kernel(dp)+self.apply(dp,0)[0]
        action.response_batch=lambda batch:np.array([action(p) for p in batch])
        return action
    def register_density_span(self,densities):
        densities=np.asarray(densities);vectors=np.array([self.pack(p) for p in densities]).T;pinverse=np.linalg.pinv(vectors);matrices=[np.zeros((len(densities),len(densities))) for unused in range(2)]
        for phi,weight,charge,spin in self.chunks:
            rho=np.array([self.rho(phi,p) for p in densities])
            for channel,fx in enumerate([charge,spin]):matrices[channel]+=np.einsum('aig,ijg,bjg,g->ab',rho,fx,rho,weight,optimize=True)
        for channel in range(2):matrices[channel]-=vectors.T@(self.ri[channel]@vectors)
        self.span=vectors,pinverse,matrices
    def spin_pair_bilinear(self,left,right,direction):
        from ._mixed_jet import MixedJet
        if left.n or right.n:raise NotImplementedError('quadrature spin recoupling nuclear jets not yet bound')
        a=self.unpack(left.v);b=self.unpack(right.v)
        cross=(self.apply(a+b,1,potential=False)[1]-self.apply(a,1,potential=False)[1]-self.apply(b,1,potential=False)[1])
        return MixedJet.constant(float(left.v@self.reference.pair_spin_hartree@right.v)+cross,0)


class _Spin:
    def __init__(self,parent):self.parent=parent;self.base=parent.base.polarization
    def __getattr__(self,key):return getattr(self.base,key)
    def evaluate(self,p):
        f,e,m=self.base.evaluate(p);v,energy=self.parent.apply(p[0]-p[1],1)
        return f+np.array([v,-v]),e+energy,m
    def response_batch(self,p):
        result=self.base.response_batch(p)
        for k,density in enumerate(p):
            v=self.parent.apply(density[0]-density[1],1)[0];result[k]+=np.array([v,-v])
        return result
