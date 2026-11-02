"""Same-geometry, same-state full DFT ROKS; never reads excitation labels."""
import socket,runpy,argparse,json,time,hashlib,fcntl
from pathlib import Path
import numpy as np
from scipy.linalg import eigh,null_space
from pyscf import gto,dft,scf
from pyscf.data.elements import ELEMENTS
from threadpoolctl import threadpool_limits
assert socket.getfqdn().lower()=='mac-2.homenet.telecomitalia.it'
BASE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101');root=Path(__file__).resolve().parent
runpy.run_path(str(BASE/'source/qualification/remote_bootstrap.py'))
from matrix_pyscf import optimize_roks
p=argparse.ArgumentParser();p.add_argument('identity');p.add_argument('functional');a=p.parse_args()
protocol=json.loads((root/'protocol.json').read_text());case=next(x for x in protocol['cases'] if x['id']==a.identity)
assert a.functional in protocol['functionals']
cohort=case['cohort'];name=case['name'];source=BASE/'oracle/MOUSE-cam-response-20261101';pred=source/'data'/cohort/'predictions'/name
folder=root/'vertical'/a.identity/a.functional;folder.mkdir(parents=True,exist_ok=True)
# Serialize identical cases across resumed/orphaned orchestration processes.
case_lock=(folder/'run.lock').open('a');fcntl.flock(case_lock,fcntl.LOCK_EX)
if (folder/'result.json').exists():print('REUSE',a.identity,a.functional);raise SystemExit(0)
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
with threadpool_limits(2):
 start=time.monotonic();paths=[source/'data'/cohort/'references'/name/'reference.npz',pred/'native_ground.npz',pred/f'native_transition_{a.identity}.npz',pred/'full_response/anchor/embedding.npz']
 with np.load(paths[0]) as z:znum=z['atomic_numbers'];xyz=z['coordinates_bohr'];ground_c=z['coefficients'].copy();ground_o=z['occupations'].copy();archived_overlap=z['overlap'].copy()
 archived_mol=scf.chkfile.load_mol(str(paths[0].with_name('backend.chk')))
 paths.append(paths[0].with_name('backend.chk'))
 assert np.max(abs(archived_mol.intor('int1e_ovlp')-archived_overlap))<1e-8
 assert np.max(abs(archived_mol.atom_coords()-xyz))<1e-8
 mol=gto.M(atom=[(ELEMENTS[int(z)],tuple(r)) for z,r in zip(znum,xyz)],unit='Bohr',basis=protocol['basis'],verbose=0)
 with np.load(paths[1]) as z:
  native_s=z['result_overlap'];native_c=eigh(z['result_fock'][0],native_s)[1];no=int(z['state_electron_count'])//2
 if paths[2].exists():
  with np.load(paths[2]) as z:target=z['amplitude']
 else:
  frozen=BASE/'oracle/MOUSE-closing-validation-20261101/disentangled_bath_spectral_block_reference_eta_0.600000/closing_blind'/name
  manifest=json.loads((frozen/'model.json').read_text());index=manifest['state_ids'].index(a.identity)
  with np.load(frozen/'arrays.npz') as z:target_c=z['states'][index]
  target=np.outer(native_c[:,:no].T@native_s@target_c[:,0],native_c[:,no:].T@native_s@target_c[:,no])
  paths[2]=frozen/'arrays.npz'
 with np.load(paths[3]) as z:lift=z['embedding']
 u,s,vh=np.linalg.svd(target,full_matrices=False);hole=lift@native_c[:,:no]@u[:,0];particle=lift@native_c[:,no:]@vh[0]
 if archived_mol.cart!=mol.cart or ground_c.shape[0]!=mol.nao_nr():
  hole=scf.addons.project_mo_nr2nr(archived_mol,hole[:,None],mol)[:,0];particle=scf.addons.project_mo_nr2nr(archived_mol,particle[:,None],mol)[:,0]
  occupied=ground_o>0;ground_c=scf.addons.project_mo_nr2nr(archived_mol,ground_c[:,occupied],mol);ground_o=ground_o[occupied]
 mf=dft.RKS(mol);mf.xc=a.functional;mf.grids.level=protocol['grids_level'];mf.small_rho_cutoff=0;mf.conv_tol=1e-11;mf.conv_tol_grad=1e-8;mf.max_cycle=150
 ground_path=folder/'ground.npz';ground_calls=0;reference_reused=False
 if ground_path.exists():
  with np.load(ground_path) as z:c=z['coefficients'].copy();energy0=float(z['energy']);mo_o=z['occupations'].copy()
 else:
  if a.functional=='CAM-B3LYP':
   archived=source/cohort/name/'CAM_full_reference/reference.npz'
   if archived.exists() and archived_mol.cart==mol.cart:
    with np.load(archived) as z:ground_c=z['coefficients'].copy();ground_o=z['occupations'].copy()
  dm=(ground_c*ground_o)@ground_c.T;f=mf.get_hcore()+mf.get_veff(dm=dm);sao=mf.get_ovlp();r=f@dm@sao-sao@dm@f
  # Verify rather than rerun an already completed, identical reference.
  if a.functional in ('PBE0','CAM-B3LYP') and ground_c.shape==(mol.nao_nr(),mol.nao_nr()) and np.max(abs(r))<2e-7:
   c=ground_c;mo_o=ground_o;energy0=float(mf.energy_tot(dm=dm));reference_reused=True
  else:
   mf.chkfile=str(folder/'ground.chk');mf.kernel(dm0=dm);ground_calls=1
   if not mf.converged:raise RuntimeError('ground reference not converged')
   c=mf.mo_coeff;mo_o=mf.mo_occ;energy0=float(mf.e_tot)
  np.savez_compressed(ground_path,coefficients=c,occupations=mo_o,energy=energy0)
 nocc=int(np.count_nonzero(mo_o));sao=mf.get_ovlp();transformed=[];weights=[]
 for block,target,last in [(c[:,:nocc],hole,True),(c[:,nocc:],particle,False)]:
  v=block.T@sao@target;norm=float(np.linalg.norm(v));weights.append(norm)
  if norm<.7:raise RuntimeError(('unresolved frozen target sector',norm))
  v/=norm;other=null_space(v[None,:]);rotation=np.column_stack([other,v]) if last else np.column_stack([v,other]);transformed.append(block@rotation)
 initial=np.column_stack(transformed);occ=np.array([mo_o/2]*2);occ[0,nocc-1]=0;occ[0,nocc]=1
 excited=dft.UKS(mol);excited.xc=a.functional;excited.grids.level=protocol['grids_level'];excited.small_rho_cutoff=0
 checkpoint=folder/'accepted.npz';seed=initial;resume=checkpoint if checkpoint.exists() else None
 # A preserved unfinished checkpoint can switch to a qualified continuation
 # seed; completed roots above are immutable and are never recalculated.
 saved_status=None
 if resume is not None:
  with np.load(resume) as z:details=json.loads(str(z['metadata']))
  saved_status=(details.get('solver_certificate') or {}).get('status')
 if saved_status!='converged':
  for source_functional in ['WB97X','CAM-B3LYP','PBE0']:
   if source_functional==a.functional:continue
   other=folder.parent/source_functional
   if not (other/'result.json').exists():continue
   seed_path=other/'qualified.npz' if (other/'qualified.npz').exists() else other/'accepted.npz'
   with np.load(seed_path) as z:seed=z['coefficients'].copy()
   write(folder/'continuation_seed.json',dict(source_functional=source_functional,sha256=digest(seed_path),preserved_checkpoint=str(resume) if resume else None))
   checkpoint=folder/('continued_'+source_functional+'.npz');resume=checkpoint if checkpoint.exists() else None
   break
 result=optimize_roks(excited,seed,occ,reference_coefficients=initial,ground_energy_hartree=energy0,
    orbital_tolerance=1e-8,maximum_iterations=120,maximum_step=.3,minimum_reference_overlap=.1,
    restart=resume,checkpoint=checkpoint,strict=False)
 hscore=float(abs(initial[:,nocc-1]@sao@result.common_coefficients[:,nocc-1]));pscore=float(abs(initial[:,nocc]@sao@result.common_coefficients[:,nocc]))
 if not result.converged or min(hscore,pscore)<.7:
  # Preserve the stationary alternative; retry the same energy using a
  # label-independent, closed-orbital preparation of the original target.
  result.save_checkpoint(folder/'alternative_stationary.npz')
  result=optimize_roks(excited,initial,occ,ground_energy_hartree=energy0,
      orbital_tolerance=1e-8,maximum_iterations=120,maximum_step=.3,
      minimum_reference_overlap=.1,initialize_closed_orbitals=True,
      checkpoint=folder/'prepared_target.npz',strict=False)
  hscore=float(abs(initial[:,nocc-1]@sao@result.common_coefficients[:,nocc-1]));pscore=float(abs(initial[:,nocc]@sao@result.common_coefficients[:,nocc]))
  if not result.converged or min(hscore,pscore)<.7:
   for source_functional in ['WB97X','CAM-B3LYP','PBE0']:
    if source_functional==a.functional:continue
    other=folder.parent/source_functional
    if not (other/'result.json').exists():continue
    seed_path=other/'qualified.npz' if (other/'qualified.npz').exists() else other/'accepted.npz'
    with np.load(seed_path) as z:seed=z['coefficients'].copy()
    # Functional continuation transfers orbitals only, never energies or
    # physical model parameters. The target and final functional stay fixed.
    result=optimize_roks(excited,seed,occ,reference_coefficients=initial,
       ground_energy_hartree=energy0,orbital_tolerance=1e-8,
       maximum_iterations=120,maximum_step=.3,minimum_reference_overlap=.1,
       checkpoint=folder/('continuation_'+source_functional+'.npz'),strict=False)
    hscore=float(abs(initial[:,nocc-1]@sao@result.common_coefficients[:,nocc-1]));pscore=float(abs(initial[:,nocc]@sao@result.common_coefficients[:,nocc]))
    write(folder/'continuation_receipt.json',dict(source_functional=source_functional,source_checkpoint=str(seed_path),sha256=digest(seed_path),hole_overlap=hscore,particle_overlap=pscore,energy_selection=False))
    if result.converged and min(hscore,pscore)>=.7:break
   if not result.converged or min(hscore,pscore)<.7:raise RuntimeError(('unqualified stationarity or frozen valence orbital identity lost after deterministic continuation',hscore,pscore))
 result.save_checkpoint(folder/'qualified.npz')
 row=dict(**case,functional=a.functional,basis=protocol['basis'],excitation_ev=result.excitation_energy_hartree*27.211386245988,
     energy_hartree=result.energy_hartree,ground_energy_hartree=energy0,spin_squared=result.spin_squared,
     orbital_residual=result.orbital_residual,iterations=result.iterations,reference_overlap=result.reference_overlap,
     hole_overlap=hscore,particle_overlap=pscore,target_sector_projection=weights,solver=result.solver_certificate,
     new_ground_SCF_calls=ground_calls,archived_ground_reused=reference_reused,target_excitation_labels_read=False,
     source_sha256={str(x):digest(x) for x in paths},cartesian_AO_reference=bool(archived_mol.cart),cartesian_AO_calculation=bool(mol.cart),seconds=time.monotonic()-start)
 write(folder/'result.json',row);print(json.dumps(row),flush=True)
