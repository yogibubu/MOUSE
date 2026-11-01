"""Optional PySCF reconstruction of native GFN2 s/p/d Gaussian orbitals.

The established s/p reconstruction is reused verbatim. For d shells, native
and libcint Cartesian conventions are reconciled by explicit AO values and
then certified against the complete native overlap matrix. Preparation only;
this module is never imported by the portable MOUSE runtime.
"""
from prepare_fresh import source_molecule as legacy_source_molecule
import numpy as np
from matrix_gfn2.basis import _primitive_normalization
from pyscf import gto
from pyscf.data.elements import ELEMENTS
ANGULAR={1:(0,0,0),2:(1,0,0),3:(0,1,0),4:(0,0,1),5:(2,0,0),6:(0,2,0),7:(0,0,2),8:(1,1,0),9:(1,0,1),10:(0,1,1)}
ORDER={0:[1],1:[2,3,4],2:[5,8,9,6,10,7]}
def source_molecule(state, *, return_certificate=False):
 b=state.basis
 if np.max(b.cartesian_type)<=4 and all(int(z) in [1,6,7,8] for z in state.atomic_numbers):
  try:
   mol, transform = legacy_source_molecule(state)
   certificate = dict(reconstruction="legacy_sp", raw_overlap_max_error=float(np.max(abs(transform.T@mol.intor('int1e_ovlp')@transform-state.overlap))), metric_calibration=False)
   return (mol, transform, certificate) if return_certificate else (mol, transform)
  except AssertionError:pass
 basis={};atoms=[];pyrows={};offset=0
 for atom,z in enumerate(state.atomic_numbers):
  label=ELEMENTS[int(z)]+str(atom);atoms.append((label,state.coordinates_bohr[atom]));shells=[]
  for representative,l in [(1,0),(2,1),(5,2)]:
   ids=np.flatnonzero((b.cartesian_atom==atom)&(b.cartesian_type==representative))
   if not len(ids):continue
   assert len(ids)==1,'Multiple radial shells need explicit shell identifiers'
   lo,hi=b.primitive_offsets[ids[0]:ids[0]+2];shells.append([l]+[[float(e),float(c)] for e,c in zip(b.primitive_exponents[lo:hi],b.primitive_coefficients[lo:hi])])
   for typ in ORDER[l]:pyrows[atom,typ]=offset;offset+=1
  basis[label]=shells
 mol=gto.M(atom=atoms,basis=basis,unit='Bohr',cart=True,verbose=0);assert mol.nao_nr()==len(b.cartesian_type)
 points=np.vstack([center+delta for center in state.coordinates_bohr for delta in [[.31,.43,.57],[-.47,.29,.61],[.53,-.37,.23]]]);py=mol.eval_gto('GTOval_cart',points);native=np.empty((len(points),len(b.cartesian_type)));transform=np.zeros((mol.nao_nr(),b.nao))
 for j,(atom,typ) in enumerate(zip(b.cartesian_atom,b.cartesian_type)):
  lo,hi=b.primitive_offsets[j:j+2];d=points-state.coordinates_bohr[atom];radial=np.exp(-np.einsum('ij,ij->i',d,d)[:,None]*b.primitive_exponents[lo:hi])@(b.primitive_coefficients[lo:hi]*np.array([_primitive_normalization(float(e),ANGULAR[int(typ)]) for e in b.primitive_exponents[lo:hi]]));native[:,j]=radial*np.prod(d**np.array(ANGULAR[int(typ)])[None,:],axis=1);p=pyrows[int(atom),int(typ)];k=int(np.argmax(abs(py[:,p])));scale=native[k,j]/py[k,p];assert np.max(abs(native[:,j]-scale*py[:,p]))<1e-10;transform[p]=scale*b.spherical_to_cartesian[j]
 raw_overlap = transform.T @ mol.intor('int1e_ovlp') @ transform
 error = float(np.max(abs(raw_overlap-state.overlap)))
 ao_error = float(np.max(abs(py@transform-native@b.spherical_to_cartesian)))
 assert ao_error < 1e-10, ('Primitive AO reconstruction mismatch', ao_error)
 # Reconcile the saved native overlap with the unscreened libcint metric.
 # Whiten the raw embedding and impose the saved native metric, explicitly.
 # This is a numerical metric reconciliation, never an excitation-energy fit.
 from scipy.linalg import eigh
 w, u = eigh(raw_overlap); wn, un = eigh(state.overlap)
 assert min(w.min(), wn.min()) > 1e-10, 'Embedding metric must be positive definite'
 calibration = (u * w**-.5) @ u.T @ (un * wn**.5) @ un.T
 change = float(np.linalg.norm(calibration-np.eye(len(w)), ord=2))
 assert change < .01, ('Metric reconciliation changes the orbital basis by >=1%', change)
 transform = transform @ calibration
 corrected_error = float(np.max(abs(transform.T@mol.intor('int1e_ovlp')@transform-state.overlap)))
 assert corrected_error < 1e-9
 certificate = dict(reconstruction='explicit_normalized_cartesian_spd', primitive_AO_max_error=ao_error, raw_overlap_max_error=error, metric_calibration=True, metric_calibration_operator_norm=change, corrected_overlap_max_error=corrected_error, minimum_raw_overlap_eigenvalue=float(w.min()), minimum_native_overlap_eigenvalue=float(wn.min()))
 return (mol, transform, certificate) if return_certificate else (mol, transform)
