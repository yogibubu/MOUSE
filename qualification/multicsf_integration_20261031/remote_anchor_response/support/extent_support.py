"""GFN2 Gaussian AO reconstruction and exact central moments; remote only."""
from remote_bootstrap import HOST
import numpy as np
from pyscf import gto

def source_molecule(state):
 b=state.basis;basis={};atoms=[];syms={1:'H',6:'C',7:'N',8:'O'}
 for atom,z in enumerate(state.atomic_numbers):
  label=f'{syms[int(z)]}{atom}';atoms.append((label,state.coordinates_bohr[atom]));shells=[]
  for ao in np.flatnonzero(b.cartesian_atom==atom):
   typ=int(b.cartesian_type[ao])
   if typ not in [1,2]:continue
   lo,hi=b.primitive_offsets[ao:ao+2]
   shells.append([0 if typ==1 else 1]+[[float(e),float(d)] for e,d in zip(b.primitive_exponents[lo:hi],b.primitive_coefficients[lo:hi])])
  basis[label]=shells
 mol=gto.M(atom=atoms,basis=basis,unit='Bohr',cart=True,verbose=0)
 t=b.spherical_to_cartesian
 assert np.max(abs(t.T@mol.intor('int1e_ovlp')@t-state.overlap))<1e-9
 return mol,t

def radii(mol,c):
 position=mol.intor('int1e_r',comp=3);second=mol.intor('int1e_r2')
 center=np.einsum('mi,dmn,ni->id',c,position,c,optimize=True)
 variance=np.einsum('mi,mn,ni->i',c,second,c,optimize=True)-np.sum(center**2,axis=1)
 if np.min(variance)<=0:raise ValueError('nonpositive central spread')
 return np.sqrt(variance)
