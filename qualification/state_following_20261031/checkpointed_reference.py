"""Reference-response scheduling amendment: fixed public symmetries/ordinals only.
Same PBE0/aug-cc-pVDZ Hamiltonian and convergence thresholds; no TBE input.
Completed/partial reference spectra are retained unchanged.
"""
from pathlib import Path
import socket,os,json,hashlib,sys,time,re
assert socket.getfqdn()=='oracle.sns.it'
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
P=Path('/home/enzo/MOUSE-publication-validation-20261030');R=Path('/home/enzo/MOUSE-state-following-20261031');sys.path.insert(0,str(P))
import blind_reference as original
from blind_mouse import normsym
from reference_eigensolver import checkpointed_solver

def run(name):
 import numpy as np
 from pyscf import gto,dft,lib,symm
 from pyscf.tdscf import rhf
 base_eigensolver=rhf.lr_eigh
 lib.num_threads(2);manifest=json.loads((P/'blind/manifest.json').read_text());items=[r for r in manifest['states'] if r['molecule']==name];row=items[0];d=P/'blind/references'/name;d.mkdir(exist_ok=True)
 if (d/'result.json').exists():return
 catalog=json.loads((P/'blind/assignment_catalog.json').read_text());targets=[r for r in catalog['states'] if r['molecule']==name];xyz=P/'blind/geometries'/f"{row['id']}.xyz";assert original.sha(xyz)==row['geometry_sha256']
 request=dict(plan_sha256=original.sha(P/'blind/references/protocol.json'),geometry_sha256=original.sha(xyz));requesthash=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
 ground=d/'reference.npz';start=time.perf_counter()
 if ground.exists():
  from pyscf import scf
  mol=scf.chkfile.load_mol(str(d/'backend.chk'));mol.max_memory=24000
  mf=dft.RKS(mol);mf.xc='PBE0';mf.grids.level=3;mf.small_rho_cutoff=0;mf.conv_tol=1e-10;mf.max_cycle=150
  with np.load(ground) as f:
   assert str(f['request_hash'])==requesthash
   mf.mo_coeff=f['coefficients'];mf.mo_occ=f['occupations'];mf.mo_energy=f['orbital_energies'];mf.e_tot=float(f['energy_hartree']);mf.converged=True
 else:
  lines=xyz.read_text().splitlines();n=int(lines[0]);mol=gto.M(atom='\n'.join(lines[2:2+n]),unit='Angstrom',basis='aug-cc-pVDZ',charge=0,spin=0,symmetry=True,verbose=4,output=str(d/'targeted_scf.log'),max_memory=24000)
  mf=dft.RKS(mol);mf.xc='PBE0';mf.grids.level=3;mf.small_rho_cutoff=0;mf.conv_tol=1e-10;mf.max_cycle=150;mf.chkfile=str(d/'backend.chk');mf.kernel()
  if not mf.converged:
   old=mf;mf=mf.newton();mf.max_cycle=100;mf.kernel(old.mo_coeff,old.mo_occ)
  if not mf.converged:raise RuntimeError('Ground DFT SCF did not converge')
  temp=d/'reference.tmp.npz';np.savez_compressed(temp,coefficients=mf.mo_coeff,occupations=mf.mo_occ,orbital_energies=mf.mo_energy,overlap=mf.get_ovlp(),energy_hartree=mf.e_tot,atomic_numbers=mol.atom_charges(),coordinates_bohr=mol.atom_coords(),request_hash=requesthash);temp.replace(ground)
  original.write(d/'ground_result.json',dict(request=request,host=socket.getfqdn(),seconds=time.perf_counter()-start,nao=mol.nao_nr(),converged=True,point_group=mol.groupname,reference_amendment='targeted response only'))
 mol.stdout=open(d/'targeted_tda.log','a');mf.verbose=4
 groups={}
 for t in targets:
  wanted=normsym(t['state_label']);groups[wanted]=max(groups.get(wanted,0),t['ordinal_same_singlet_symmetry'])
 vectors=[];energies=[];conv=[];labels=[];attempts=[];unresolved=[]
 for wanted,count in sorted(groups.items()):
  try:symm.irrep_name2id(mol.groupname,wanted)
  except Exception:
   unresolved.append(dict(symmetry=wanted,reason='Point-group correlation not implemented; no guessed assignment'));continue
  cache=d/f'targeted_{wanted.replace(chr(34),"doubleprime").replace(chr(39),"prime")}.npz'
  if cache.exists():
   with np.load(cache) as f:
    xs=f['amplitudes'].copy();es=f['energies_hartree'].copy();cs=f['converged'].copy()
    valid=('request_hash' not in f.files or str(f['request_hash'])==requesthash) and len(cs)>=count and all(cs)
   if valid:
    vectors.extend(xs);energies.extend(map(float,es));conv.extend(map(bool,cs));labels.extend([wanted]*len(cs));attempts.append(dict(symmetry=wanted,requested_roots=count,returned_roots=len(cs),converged=int(sum(cs)),reused_completed_checkpoint=True));continue
  signature=hashlib.sha256(json.dumps(dict(ground_sha256=original.sha(ground),symmetry=wanted,roots=count,request_hash=requesthash,xc='PBE0',basis='aug-cc-pVDZ',grid_level=3,tolerance=1e-7,lindep=1e-18),sort_keys=True).encode()).hexdigest()
  folder=R/'reduced_validation/reference_ritz'/name/wanted.replace(chr(34),'doubleprime').replace(chr(39),'prime')
  rhf.lr_eigh=checkpointed_solver(base_eigensolver,folder,signature)
  td=mf.TDA();td.wfnsym=wanted;td.nstates=count;td.conv_tol=1e-7;td.lindep=1e-18;td.max_cycle=150;td.max_memory=24000;t0=time.perf_counter();td.kernel()
  for (x,y),e,c in zip(td.xy,td.e,td.converged):vectors.append(x.ravel());energies.append(float(e));conv.append(bool(c));labels.append(wanted)
  attempts.append(dict(symmetry=wanted,requested_roots=count,returned_roots=len(td.e),converged=int(sum(td.converged)),seconds=time.perf_counter()-t0))
  # Save every finished symmetry immediately for numerical audit and continuation.
  np.savez_compressed(d/f'targeted_{wanted.replace(chr(34),"doubleprime").replace(chr(39),"prime")}.npz',amplitudes=np.array([x.ravel() for x,y in td.xy]),energies_hartree=np.asarray(td.e),converged=np.asarray(td.converged),request_hash=requesthash)
 order=np.argsort(energies);nocc=int(np.count_nonzero(mf.mo_occ==2));nv=int(np.count_nonzero(mf.mo_occ==0));a=np.array(vectors).reshape(-1,nocc*nv)
 temp=d/'backend_tda.tmp.npz';np.savez_compressed(temp,amplitudes=a[order],energies_hartree=np.array(energies)[order],converged=np.array(conv,dtype=bool)[order],state_symmetries=np.array(labels,dtype=str)[order],request_hash=requesthash);temp.replace(d/'backend_tda.npz')
 states=[dict(root=i+1,symmetry=labels[k],energy_ev=energies[k]*27.211386245988,converged=conv[k]) for i,k in enumerate(order)]
 result=dict(status='reference_ready' if conv and all(conv) and not unresolved else 'reference_partial',request=request,host=socket.getfqdn(),molecule=name,manifest_states=[x['id'] for x in items],nao=mol.nao_nr(),nocc=nocc,nvirtual=nv,point_group=mol.groupname,seconds_current_run=time.perf_counter()-start,roots=len(energies),converged_roots=sum(conv),covers_20_ev=False,window_policy='Only fixed public target symmetries and ordinal roots, not a complete 20 eV spectrum',states=states,attempts=attempts,unresolved_symmetries=unresolved,original_ground_checkpoint_preserved=True,reference_labels_read=False,amendment_protocol_sha256=original.sha(P/'blind/targeted_protocol.json'),reference_eigensolver_amendment_sha256=original.sha(R/'reduced_validation/reference_solver_amendment.json'))
 original.write(d/'result.json',result);print(name,result['status'],len(states),flush=True)
if __name__=='__main__':run(sys.argv[1])
