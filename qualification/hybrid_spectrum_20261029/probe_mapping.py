from remote_bootstrap import HOST
from pathlib import Path
import json,numpy as np
from pyscf import gto,scf,lib
from scipy.optimize import linear_sum_assignment
from matrix_gfn2.deltascf_derivatives import load_derivative_point
lib.num_threads(2)
PREVIOUS=Path('/home/enzo/MOUSE-deltascf-validation-20261029/uracil_uridine_20261029')
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
small=gto.M(atom=atoms,basis=basis,unit='Bohr',cart=True,verbose=0)
t=b.spherical_to_cartesian
error=np.max(abs(t.T@small.intor('int1e_ovlp')@t-s.overlap));print('GFN2 metric discrepancy',error,flush=True);assert error<1e-9
out=Path(__file__).resolve().parent/'results';out.mkdir(exist_ok=True)
root=Path('/home/enzo/MATRIX-runs/tda-uracil-application-20261022-v1/docs/reports/tda_uracil_application_20261022/results')
for name in ['b3lyp_avdz','pbe0_avdz','pbe0_avtz']:
 folder=root/name;mol=scf.chkfile.load_mol(str(folder/'backend.chk'))
 with np.load(folder/'reference.npz') as f:c=f['coefficients'];e=f['orbital_energies_hartree'];o=f['occupations'];ov=f['overlap']
 assert np.max(abs(mol.atom_coords()-s.coordinates_bohr))<1e-8
 assert np.max(abs(mol.intor('int1e_ovlp')-ov))<1e-9
 cross=gto.intor_cross('int1e_ovlp',mol,small)@t
 mo=c.T@cross@s.coefficients
 pairs=[]
 for occupation in (2,0):
  si=np.flatnonzero(s.alpha_occupations+s.beta_occupations==occupation);di=np.flatnonzero(o==occupation)
  rows,cols=linear_sum_assignment(-abs(mo[np.ix_(di,si)]).T**2)
  for r,k in zip(rows,cols):pairs.append(dict(se=int(si[r]),dft=int(di[k]),overlap=float(abs(mo[di[k],si[r]])),gap_energy=float(e[di[k]])))
 pairs=sorted(pairs,key=lambda d:d['se'])
 np.savez_compressed(out/f'{name}_mapping_probe.npz',cross_overlap=cross,mo_overlap=mo,dft_energies=e,dft_occupations=o,se_energies=s.orbital_energies_hartree,se_occupations=s.alpha_occupations+s.beta_occupations)
 (out/f'{name}_mapping_probe.json').write_text(json.dumps(dict(host=HOST,metric_error=float(error),pairs=pairs),indent=2))
 print(name,'frontier',[(d['se'],d['dft'],round(d['overlap'],3),round(d['gap_energy']*27.211386,3)) for d in pairs if 15<=d['se']<=25],flush=True)
