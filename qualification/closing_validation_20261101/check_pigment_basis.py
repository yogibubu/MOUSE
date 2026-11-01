from gaussian_embedding_basis import *
from prepare_fresh import HERE,write
from matrix_gfn2.electronic_state import GFN2OrbitalState,GFN2OrbitalBasis
from dataclasses import fields
for name in ['Beta_carotene','Chlorophyllide_a']:
 folder=HERE/'pigment_preflight'/name
 with np.load(folder/'native_ground.npz') as z:
  basis=GFN2OrbitalBasis(**{f.name:z['basis_'+f.name].copy() for f in fields(GFN2OrbitalBasis)});kw={f.name:z[f.name].copy() for f in fields(GFN2OrbitalState) if f.name!='basis'}
 for field in ['model_id']:kw[field]=str(kw[field])
 kw['ao_shell']=tuple(kw['ao_shell'].tolist())
 for field in ['electron_count','charge','multiplicity']:kw[field]=int(kw[field])
 kw['total_energy_hartree']=float(kw['total_energy_hartree']);s=GFN2OrbitalState(**kw,basis=basis);mol,t,certificate=source_molecule(s,return_certificate=True);error=float(np.max(abs(t.T@mol.intor('int1e_ovlp')@t-s.overlap)));write(folder/'embedding_basis_certificate.json',dict(molecule=name,spherical_native_orbitals=s.basis.nao,cartesian_functions=mol.nao_nr(),maximum_angular_momentum=int(max(mol.bas_angular(i) for i in range(mol.nbas))),overlap_max_error=error,embedding=certificate));print(name,error,flush=True)
