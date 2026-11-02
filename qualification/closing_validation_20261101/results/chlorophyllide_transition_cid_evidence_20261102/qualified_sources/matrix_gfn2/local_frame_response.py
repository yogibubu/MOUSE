"""Frozen AO response transported by static covalent-neighborhood frames.

Each atomic p shell follows its own fixed local cluster. The analytic Kabsch
rotation jets include every first and second nuclear derivative. Cluster
membership is chosen once at the anchor and is part of the saved Hamiltonian.
"""
from dataclasses import dataclass, fields
from hashlib import sha256
import numpy as np
from .nonlocal_response import FrozenNonlocalResponseReference, PairSpin, PairTerm, rotation_jet
from .spin_reference import FrozenResponseReference
from .calculator import ANGSTROM_TO_BOHR


def covalent_frame_clusters(reference, *, heavy_first=False):
    from matrix_chem.topology.covalent_radii import covalent_radius
    xyz=reference.anchor_coordinates_bohr
    radii=[covalent_radius(int(z)) for z in reference.atomic_numbers]
    if any(r is None or r<=0 for r in radii):
        raise ValueError('covalent radii unavailable for a local response frame')
    adjacency=[set() for _ in xyz]
    for i in range(len(xyz)):
        for j in range(i):
            if np.linalg.norm(xyz[i]-xyz[j])/ANGSTROM_TO_BOHR<=radii[i]+radii[j]+.28:
                adjacency[i].add(j);adjacency[j].add(i)
    result=np.full((len(xyz),len(xyz)),-1,dtype=np.int64)
    for atom in range(len(xyz)):
        modes=[True,False] if heavy_first and reference.atomic_numbers[atom]>1 else [False]
        found=False
        for heavy_only in modes:
            selected={atom};frontier={atom}
            for _ in xyz:
                frontier={j for i in frontier for j in adjacency[i]
                    if not heavy_only or reference.atomic_numbers[j]>1}-selected
                selected|=frontier
                ids=sorted(selected);points=xyz[ids]-xyz[ids].mean(axis=0)
                singular=np.linalg.svd(points,compute_uv=False)
                if len(singular)>=2 and singular[1]>1e-7:
                    result[atom,:len(ids)]=ids;found=True;break
                if not frontier:break
            if found:break
        if not found:raise ValueError('no noncollinear covalent frame for atom '+str(atom))
    return result


class LocalTransportedOperators:
 def __init__(self,state,anchor,operators,clusters):
  self.state=state;self.anchor=anchor;self.original=operators;self.clusters=tuple(tuple(int(i) for i in row if i>=0) for row in clusters);self.blocks=[]
  for atom in range(len(state.atomic_numbers)):
   for label in dict.fromkeys(state.ao_shell):
    ids=[i for i,(a,s) in enumerate(zip(state.ao_atom,state.ao_shell)) if a==atom and s==label]
    if not ids or str(label).endswith('s') and len(ids)==1:continue
    if len(ids)!=3 or not str(label).endswith('p'):raise ValueError('local frame prototype supports s/p shells')
    self.blocks.append((atom,ids))
  n=operators.shape[-1];nr=state.coordinates_bohr.size;self.u=np.eye(n);self.du=np.zeros((nr,n,n));self.jets={}
  for atom,ids in self.blocks:
   cluster=np.array(self.clusters[atom],dtype=int);r,dr,_=rotation_jet(state.coordinates_bohr[cluster],anchor[cluster],second=False)
   coordinates=(3*cluster[:,None]+np.arange(3)).ravel();self.jets[atom]=dict(cluster=cluster,coordinates=coordinates,dr=dr,d2=None)
   self.u[np.ix_(ids,ids)]=r
   for k,coordinate in enumerate(coordinates):self.du[coordinate][np.ix_(ids,ids)]=dr[k]
  self.a=self.u@operators@self.u.T;self._first=None
 def frame_jet(self,*,second=False):
  """Small local AO rotation jets; feature operators are not differentiated."""
  if second:
   for jet in self.jets.values():
    if jet['d2'] is None:
     cluster=jet['cluster'];_,_,jet['d2']=rotation_jet(self.state.coordinates_bohr[cluster],self.anchor[cluster])
  return self.du,None
 def frame_curvature(self,adjoint):
  self.frame_jet(second=True);nr=self.state.coordinates_bohr.size;value=np.zeros((nr,nr))
  for atom,ids in self.blocks:
   jet=self.jets[atom];coords=jet['coordinates']
   value[np.ix_(coords,coords)]+=np.einsum('rsij,ij->rs',jet['d2'],adjoint[np.ix_(ids,ids)])
  return value
 def first(self):
  if self._first is None:self._first=self.du[:,None]@self.original@self.u.T+self.u@self.original@self.du[:,None].transpose(0,1,3,2)
  return self._first
 def second(self,i,j):
  d2u=np.zeros_like(self.u)
  for atom,ids in self.blocks:
   jet=self.jets[atom];coords=jet['coordinates'];ii=np.flatnonzero(coords==i);jj=np.flatnonzero(coords==j)
   if not len(ii) or not len(jj):continue
   if jet['d2'] is None:
    cluster=jet['cluster'];_,_,jet['d2']=rotation_jet(self.state.coordinates_bohr[cluster],self.anchor[cluster])
   d2u[np.ix_(ids,ids)]=jet['d2'][ii[0],jj[0]]
  return (d2u@self.original@self.u.T+self.u@self.original@d2u.T
   +self.du[i]@self.original@self.du[j].T+self.du[j]@self.original@self.du[i].T)


@dataclass(frozen=True,repr=False)
class FrozenLocalFrameResponseReference(FrozenNonlocalResponseReference):
    frame_clusters: np.ndarray = None

    def __post_init__(self):
        super().__post_init__()
        n=len(self.atomic_numbers);raw=np.asarray(self.frame_clusters)
        if (raw.shape!=(n,n) or not np.issubdtype(raw.dtype,np.integer)
                or np.any(raw < -1) or np.any(raw >= n)):
            raise ValueError('invalid covalent frame cluster table')
        clusters=np.array(raw,dtype=np.int64,copy=True)
        for atom,row in enumerate(clusters):
            ids=row[row>=0]
            if len(ids)<3 or atom not in ids or len(set(ids))!=len(ids):
                raise ValueError('local frame needs a labelled noncollinear cluster')
            rotation_jet(self.anchor_coordinates_bohr[ids],self.anchor_coordinates_bohr[ids],second=False)
        clusters.setflags(write=False);object.__setattr__(self,'frame_clusters',clusters)

    @classmethod
    def from_reference(cls,reference,*,frame_clusters=None,heavy_first=False):
        if not isinstance(reference,FrozenNonlocalResponseReference):
            raise TypeError('local frames require a full frozen AO response')
        clusters=covalent_frame_clusters(reference,heavy_first=heavy_first) if frame_clusters is None else frame_clusters
        return cls(**{f.name:getattr(reference,f.name) for f in fields(FrozenNonlocalResponseReference)},
                   frame_clusters=clusters)

    @property
    def fingerprint(self):
        return sha256(super().fingerprint.encode()+b':local_covalent_kabsch.v1:'+self.frame_clusters.tobytes()).hexdigest()

    def __repr__(self):
        return f'FrozenLocalFrameResponseReference({self.fingerprint})'

    def bind(self,state,partition):
        base=FrozenResponseReference.bind(self,state,partition)
        transport=LocalTransportedOperators(state,self.anchor_coordinates_bohr,self.operators,self.frame_clusters)
        return PairSpin(base,PairTerm(transport,self.pair_spin_hartree))

    def bind_charge(self,state,partition):
        FrozenResponseReference.bind(self,state,partition)
        transport=LocalTransportedOperators(state,self.anchor_coordinates_bohr,self.operators,self.frame_clusters)
        return PairTerm(transport,self.pair_charge_hartree,self.pair_reference_moments,self.pair_linear_hartree)
