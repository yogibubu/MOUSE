"""Separate compact-basis loss, ground-sector leakage and seed limitations."""
from pathlib import Path
import os,socket,sys,json,hashlib
assert socket.getfqdn()=='oracle.sns.it'
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MOUSE_VALIDATION_ROOT='/home/enzo/MOUSE-state-following-20261031')
R=Path('/home/enzo/MOUSE-state-following-20261031');P=Path('/home/enzo/MOUSE-publication-validation-20261030')
sys.path.insert(0,str(R/'source/qualification'));import remote_bootstrap
import numpy as np
from pyscf import scf,gto,lib
from threadpoolctl import threadpool_limits
from matrix_gfn2.deltascf_derivatives import load_derivative_point
sys.path.append('/home/enzo/MOUSE-deltascf-validation-20261029/hybrid_spectrum_20261029');from extent_support import source_molecule
lib.num_threads(2);rows=[]
with threadpool_limits(limits=2):
 for folder in sorted((P/'blind/prepared').iterdir()):
  if not (folder/'ground.npz').exists() or not (P/'blind/mouse_reference'/folder.name/'backend_tda.npz').exists():continue
  with np.load(folder/'ground.npz') as f:xyz=f['coordinates_angstrom'];request=str(f['request_hash'])
  gs=load_derivative_point(folder/'ground.npz',request_hash=request,coordinates_angstrom=xyz,require_gradient=False);s=gs.reference_state
  reference=P/'blind/mouse_reference'/folder.name
  mol=scf.chkfile.load_mol(str(reference/'backend.chk'));small,t=source_molecule(s);cross=gto.intor_cross('int1e_ovlp',mol,small)@t
  with np.load(reference/'reference.npz') as f:cd=f['coefficients'];od=f['occupations']
  oi=np.flatnonzero(s.alpha_occupations==1);vi=np.flatnonzero(s.alpha_occupations==0);di=np.flatnonzero(od==2);da=np.flatnonzero(od==0)
  overlap=s.coefficients.T@cross.T@cd;occ=overlap[:,di];virt=overlap[:,da]
  with np.load(reference/'backend_tda.npz') as f:amplitudes=f['amplitudes'].reshape(-1,len(di),len(da))
  assignment=json.loads((P/'blind/mouse'/folder.name/'reference_assignment.json').read_text())
  for target in assignment['targets']:
   x=amplitudes[target['reference_root']-1];x=x/np.linalg.norm(x)
   all_projected=occ@x@virt.T;sector=occ[oi]@x@virt[vi].T;fullnorm=float(np.linalg.norm(all_projected));norm=float(np.linalg.norm(sector));sigma=np.linalg.svd(sector,compute_uv=False)
   lead=float(sigma[0]/norm) if norm else 0.;canonical=float(np.max(abs(sector))/norm) if norm else 0.
   classification='compact_basis_limit' if fullnorm<.7 else 'ground_occupation_sector_leakage' if norm<.7 else 'multiconfigurational_target' if lead<.7 else 'canonical_seed_limitation' if canonical<.7 else 'representable_single_pair'
   row=dict(molecule=folder.name,state=target['id'],family=target['family'],reference_root=target['reference_root'],full_compact_projection_norm=fullnorm,occupied_virtual_projection_norm=norm,leading_pair_amplitude=lead,leading_pair_weight=lead**2,maximum_canonical_amplitude=canonical,classification=classification,numerical_reference_energies_used=False)
   rows.append(row);print(row,flush=True)
result=dict(host=socket.getfqdn(),scope='geometric Hilbert-space/occupation-sector projection of assigned unit-normalized DFT transitions; no new electronic optimization',rows=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),baseline_classifications_preserved=True)
(R/'representation_audit.json').write_text(json.dumps(result,indent=2)+'\n')
