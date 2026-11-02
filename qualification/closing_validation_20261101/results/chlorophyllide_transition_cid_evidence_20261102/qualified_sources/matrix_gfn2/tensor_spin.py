"""Rotation-invariant on-site sp dipole and p quadrupole spin response.

The scalar coefficient is identical within each irreducible (3 or 5 dimensional)
block. No fixed laboratory axis or state-specific energy offset is introduced.
Moments use sym(M S); the AO metric is differentiated analytically.
"""
from dataclasses import dataclass
import numpy as np


def onsite_spin_generators(state):
    n=len(state.overlap);operators=[];groups=[]
    for atom in range(len(state.atomic_numbers)):
        ids=np.flatnonzero(np.asarray(state.ao_atom)==atom)
        ss=[i for i in ids if state.ao_shell[i][-1].lower()=='s']
        pp=[i for i in ids if state.ao_shell[i][-1].lower()=='p']
        if not pp:continue
        if len(ss)!=1 or len(pp)!=3 or len(ids)!=4:
            raise ValueError('tensor spin currently supports one s and one p shell per atom')
        for p in pp:
            a=np.zeros((n,n));a[ss[0],p]=a[p,ss[0]]=1/np.sqrt(2)
            operators.append(a);groups.append((atom,0))
        for diagonal in ([1,-1,0],[1,1,-2]):
            a=np.zeros((n,n));d=np.asarray(diagonal,dtype=float);d/=np.linalg.norm(d)
            a[pp,pp]=d;operators.append(a);groups.append((atom,1))
        for i,j in [(0,1),(0,2),(1,2)]:
            a=np.zeros((n,n));a[pp[i],pp[j]]=a[pp[j],pp[i]]=1/np.sqrt(2)
            operators.append(a);groups.append((atom,1))
    return np.asarray(operators).reshape(-1,n,n),np.asarray(groups,dtype=int).reshape(-1,2)


@dataclass(frozen=True)
class TensorSpinPolarization:
    base: object
    generators: np.ndarray
    coefficients_hartree: np.ndarray

    @property
    def overlap(self):return self.base.overlap
    @property
    def orbital_shell(self):return self.base.orbital_shell
    @property
    def coupling_hartree(self):return self.base.coupling_hartree

    def operators(self,metric=None):
        s=self.overlap if metric is None else metric
        return .5*(s@self.generators+self.generators@s)

    def moments(self,m,metric=None):return np.einsum('aij,ij->a',self.operators(metric),m)

    def evaluate(self,spin_densities):
        f,e,mag=self.base.evaluate(spin_densities);m=spin_densities[0]-spin_densities[1]
        q=self.moments(m);v=self.coefficients_hartree*q
        potential=np.einsum('a,aij->ij',v,self.operators())
        return f+np.array([potential,-potential]),e+.5*float(q@v),mag

    def tensor_metric_potential(self,spin_densities):
        m=spin_densities[0]-spin_densities[1]
        a=np.einsum('a,aij->ij',self.coefficients_hartree*self.moments(m),self.generators)
        return .5*(m@a+a@m)

    def tensor_geometry(self,spin_densities,ds,d2s):
        m=spin_densities[0]-spin_densities[1];b=self.operators()
        db=np.array([self.operators(x) for x in ds]);q=self.moments(m)
        dq=np.einsum('raij,ij->ra',db,m);v=self.coefficients_hartree*q
        df=np.einsum('a,raij->rij',v,db)+np.einsum('ra,aij->rij',dq*self.coefficients_hartree,b)
        rr=(dq*self.coefficients_hartree)@dq.T
        for i in range(len(ds)):
            for j in range(len(ds)):rr[i,j]+=float(v@self.moments(m,d2s[i,j]))
        return df,rr


@dataclass(frozen=True)
class TensorChargeCorrection:
    base: object
    tensor: TensorSpinPolarization

    def evaluate(self,total_density):
        f,e=self.base.evaluate(total_density);q=self.tensor.moments(total_density)
        v=self.tensor.coefficients_hartree*q
        return f+np.einsum('a,aij->ij',v,self.tensor.operators()),e+.5*float(q@v)

    def response(self,dp):
        return self.base.response(dp)+np.einsum('a,aij->ij',self.tensor.coefficients_hartree*self.tensor.moments(dp),self.tensor.operators())

    def metric_potential(self,p):
        return self.base.metric_potential(p)+self.tensor.tensor_metric_potential(np.array([p,np.zeros_like(p)]))

    def geometry(self,p,ds,d2s):
        f,e=self.base.geometry(p,ds,d2s)
        df,de=self.tensor.tensor_geometry(np.array([p,np.zeros_like(p)]),ds,d2s)
        return f+df,e+de
