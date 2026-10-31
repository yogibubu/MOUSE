"""Measure translation-invariant orbital spreads on verified Oracle only."""
from remote_bootstrap import HOST
from pathlib import Path
import json,numpy as np
from pyscf import gto,scf,lib
from matrix_gfn2.deltascf_derivatives import load_derivative_point
lib.num_threads(2)
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results/extent';OUT.mkdir(exist_ok=True)
PREVIOUS=ROOT.parent/'uracil_uridine_20261029'
x=np.array(json.loads((PREVIOUS/'inputs/quest_s0.json').read_text())['coordinates_angstrom'])
g=load_derivative_point(PREVIOUS/'results/quest_s0/ground.npz',request_hash='quest_s0',coordinates_angstrom=x);s=g.reference_state;b=s.basis
basis={};atoms=[];syms={1:'H',6:'C',7:'N',8:'O'}
for atom,z in enumerate(s.atomic_numbers):
 label=f'{syms[int(z)]}{atom}';atoms.append((label,s.coordinates_bohr[atom]));shells=[]
 for ao in np.flatnonzero(b.cartesian_atom==atom):
  typ=int(b.cartesian_type[ao])
  if typ not in [1,2]:continue
  lo,hi=b.primitive_offsets[ao:ao+2]
  shells.append([0 if typ==1 else 1]+[[float(e),float(d)] for e,d in zip(b.primitive_exponents[lo:hi],b.primitive_coefficients[lo:hi])])
 basis[label]=shells
small=gto.M(atom=atoms,basis=basis,unit='Bohr',cart=True,verbose=0);t=b.spherical_to_cartesian
assert np.max(abs(t.T@small.intor('int1e_ovlp')@t-s.overlap))<1e-9

def moments(mol,c):
 r=mol.intor('int1e_r',comp=3);r2=mol.intor('int1e_r2')
 cent=np.einsum('mi,dmn,ni->id',c,r,c,optimize=True)
 second=np.einsum('mi,mn,ni->i',c,r2,c,optimize=True)
 variance=second-np.sum(cent**2,axis=1)
 assert np.min(variance)>0
 # Change AO moment origin algebraically and verify spread invariance.
 u=np.array([2.,-3.,1.]);ov=mol.intor('int1e_ovlp')
 rc=r-u[:,None,None]*ov
 rr=r2-2*np.einsum('d,dmn->mn',u,r)+(u@u)*ov
 c2=np.einsum('mi,dmn,ni->id',c,rc,c,optimize=True)
 v2=np.einsum('mi,mn,ni->i',c,rr,c,optimize=True)-np.sum(c2**2,axis=1)
 assert np.max(abs(v2-variance))<1e-7
 return cent,variance,np.sqrt(variance)
cent,var,radii=moments(small,t@s.coefficients)
np.savez_compressed(OUT/'gfn2_moments.npz',centroids=cent,variances=var,radii=radii)
records=[dict(model='GFN2',frontier=[dict(orbital=i,radius_bohr=float(radii[i]),centroid=cent[i].tolist()) for i in [19,20,21,22]])]
dftroot=Path('/home/enzo/MATRIX-runs/tda-uracil-application-20261022-v1/docs/reports/tda_uracil_application_20261022/results')
for name in ['b3lyp_avdz','pbe0_avdz','pbe0_avtz']:
 folder=dftroot/name;mol=scf.chkfile.load_mol(str(folder/'backend.chk'))
 with np.load(folder/'reference.npz') as d:c=d['coefficients']
 cent,var,radii=moments(mol,c)
 with np.load(ROOT/'results'/f'{name}_mapped_spectrum.npz') as f:ids=f['reference_indices']
 np.savez_compressed(OUT/f'{name}_moments.npz',centroids=cent,variances=var,radii=radii,mapped_radii=radii[ids],reference_indices=ids)
 records.append(dict(model=name,frontier=[dict(orbital=i,dft_orbital=int(ids[i]),radius_bohr=float(radii[ids[i]]),centroid=cent[ids[i]].tolist()) for i in [19,20,21,22]]))
(OUT/'moments.json').write_text(json.dumps(dict(host=HOST,records=records),indent=2)+'\n')
for row in records:print(row,flush=True)
