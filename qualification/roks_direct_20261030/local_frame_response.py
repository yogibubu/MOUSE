"""Research-only local atomic transport for the unchanged frozen AO response.
Clusters are fixed from the anchor covalent graph. No geometry re-selection,
energy fitting, offset, or nuclear finite differences enter the model.
"""
from dataclasses import fields
from hashlib import sha256
import json
import numpy as np
from matrix_gfn2.nonlocal_response import FrozenNonlocalResponseReference,PairSpin,PairTerm,rotation_jet
from matrix_gfn2.spin_reference import FrozenResponseReference
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from matrix_chem.peptide_order import infer_covalent_adjacency


def clusters_for(reference):
 xyz=reference.anchor_coordinates_bohr;atoms=[{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in reference.atomic_numbers]
 adjacency=infer_covalent_adjacency(atoms,xyz/ANGSTROM_TO_BOHR);clusters=[]
 for atom in range(len(xyz)):
  selected={atom};frontier={atom}
  for depth in range(len(xyz)):
   frontier=set(j for i in frontier for j in adjacency[i])-selected;selected|=frontier
   ids=sorted(selected);local=xyz[ids]-xyz[ids].mean(axis=0)
   singular=np.linalg.svd(local,compute_uv=False)
   if len(singular)>=2 and singular[1]>1e-7:break
   if not frontier:raise ValueError('no noncollinear local frame for atom '+str(atom))
  clusters.append(tuple(ids))
 return tuple(clusters)


class LocalTransport:
 def __init__(self,state,anchor,operators,clusters):
  self.state=state;self.anchor=anchor;self.original=operators;self.clusters=clusters;self.blocks=[]
  for atom in range(len(state.atomic_numbers)):
   for label in dict.fromkeys(state.ao_shell):
    ids=[i for i,(a,s) in enumerate(zip(state.ao_atom,state.ao_shell)) if a==atom and s==label]
    if not ids or str(label).endswith('s') and len(ids)==1:continue
    if len(ids)!=3 or not str(label).endswith('p'):raise ValueError('local frame prototype supports s/p shells')
    self.blocks.append((atom,ids))
  n=operators.shape[-1];nr=state.coordinates_bohr.size;self.u=np.eye(n);self.du=np.zeros((nr,n,n));self.jets={}
  for atom,ids in self.blocks:
   cluster=np.array(clusters[atom],dtype=int);r,dr,_=rotation_jet(state.coordinates_bohr[cluster],anchor[cluster],second=False)
   coordinates=(3*cluster[:,None]+np.arange(3)).ravel();self.jets[atom]=dict(cluster=cluster,coordinates=coordinates,dr=dr,d2=None)
   self.u[np.ix_(ids,ids)]=r
   for k,coordinate in enumerate(coordinates):self.du[coordinate][np.ix_(ids,ids)]=dr[k]
  self.a=self.u@operators@self.u.T;self._first=None
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


class LocalReference(FrozenNonlocalResponseReference):
 @classmethod
 def from_reference(cls,reference):
  result=cls(**{f.name:getattr(reference,f.name) for f in fields(FrozenNonlocalResponseReference)})
  object.__setattr__(result,'clusters',clusters_for(reference));return result
 @property
 def fingerprint(self):return sha256((super().fingerprint+':local_covalent_kabsch.v1:'+json.dumps(self.clusters)).encode()).hexdigest()
 def bind(self,state,partition):
  base=FrozenResponseReference.bind(self,state,partition)
  t=LocalTransport(state,self.anchor_coordinates_bohr,self.operators,self.clusters)
  return PairSpin(base,PairTerm(t,self.pair_spin_hartree))
 def bind_charge(self,state,partition):
  FrozenResponseReference.bind(self,state,partition)
  t=LocalTransport(state,self.anchor_coordinates_bohr,self.operators,self.clusters)
  return PairTerm(t,self.pair_charge_hartree,self.pair_reference_moments,self.pair_linear_hartree)


def local_pair_score(anchor,candidate,reference):
 vectors=[]
 opens=np.flatnonzero(anchor.occupations[0]!=anchor.occupations[1])
 for result in [anchor,candidate]:
  s=result.reference_state;values,v=np.linalg.eigh(s.overlap);root=(v*np.sqrt(values))@v.T
  t=LocalTransport(s,reference.anchor_coordinates_bohr,np.eye(len(values))[None],reference.clusters)
  vectors.append(t.u.T@root@result.common_coefficients[:,opens])
 a=vectors[0].T@vectors[1]
 return float(abs(a[0,0]*a[1,1]+a[0,1]*a[1,0]))
