"""Verified-host, checkpointed preparation of fixed independent inputs.

No QUEST excitation energies are read. The legacy modules supply the identical
CAM response, native orbital preservation and spin-pure stationary solver.
"""
from pathlib import Path
import sys,os,socket,runpy,json,hashlib,time,argparse
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
assert os.popen('scutil --get ComputerName').read().strip()=='Mac Studio di Vincenzo'
BASE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101')
HERE=BASE/'oracle/MOUSE-closing-validation-20261101';CAM=BASE/'oracle/MOUSE-cam-response-20261101';NATIVE=BASE/'oracle/MOUSE-native-reference-20261101';Q=BASE/'oracle/MOUSE-response-consistent-nosi-20261101';REPAIR=BASE/'oracle/MOUSE-pi-pi-transfer-repair-20261101'
runpy.run_path(str(BASE/'source/qualification/remote_bootstrap.py'))
sys.path.insert(0,str(CAM))
from anchor import *
from anchor import anchor
from calibrate import calibrate
from full_reference import full_reference
from matrix_gfn2.response_reference import stationary_response_reference
from matrix_gfn2 import evaluate_gfn2_deltascf
from scipy.linalg import solve
import importlib.util

def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def link(a,b):
 b.parent.mkdir(parents=True,exist_ok=True)
 if not b.exists():b.symlink_to(a.resolve())

def initial(name,identity,xyzfile,cohort='closing_blind',target_model=None):
 dest=CAM/'data'/cohort;src=dest/'predictions'/name;ref=dest/'references'/name;ref.mkdir(parents=True,exist_ok=True);src.mkdir(parents=True,exist_ok=True)
 xyztext=Path(xyzfile).read_text().splitlines();atoms='\n'.join(xyztext[2:]);mol=gto.M(atom=atoms,unit='Angstrom',basis='aug-cc-pVDZ',spin=0,symmetry=True,verbose=0)
 xyz=mol.atom_coords()/ANGSTROM_TO_BOHR
 rp=ref/'reference.npz'
 if not rp.exists():
  mf=dft.RKS(mol);mf.xc='PBE0';mf.grids.level=3;mf.small_rho_cutoff=0.;mf.conv_tol=1e-10;mf.conv_tol_grad=1e-7;mf.max_cycle=150;mf.chkfile=str(ref/'backend.chk')
  dm=None
  if (ref/'backend.chk').exists():
   try:dm=mf.init_guess_by_chkfile(str(ref/'backend.chk'))
   except Exception:pass
  mf.kernel(dm0=dm);assert mf.converged
  np.savez_compressed(rp,coefficients=mf.mo_coeff,occupations=mf.mo_occ,orbital_energies=mf.mo_energy,overlap=mf.get_ovlp(),atomic_numbers=mol.atom_charges(),coordinates_bohr=mol.atom_coords())
 with np.load(rp) as z:refdata={k:z[k].copy() for k in z.files}
 calc=GFN2Calculator([a[0] for a in mol._atom],temperature_kelvin=0.);gp=src/'native_ground.npz'
 if not gp.exists():
  g,_=stationary_response_reference(calc,xyz);s=g.orbital_state
  raw=evaluate_gfn2_deltascf(calc,xyz,s.alpha_occupations,s.beta_occupations,ground_state=g,selection='aufbau',orbital_tolerance=1e-10,density_tolerance=1e-10,energy_tolerance=1e-12,maximum_iterations=500)
  save_derivative_point(gp,raw,request_hash='closing:native',coordinates_angstrom=xyz)
 raw=loadpoint(gp);s=canonical_state(raw);n=len(s.overlap);no=s.electron_count//2
 link(gp,src/'grounds/0/stationary.npz')
 folder=src/'full_response/anchor';folder.mkdir(parents=True,exist_ok=True);ep=folder/'embedding.npz'
 if not ep.exists():
  cd=refdata['coefficients'];sd=refdata['overlap'];nc=np.count_nonzero(refdata['occupations']==2)-no;core=cd[:,:nc]
  small,t=source_molecule(s);cross=gto.intor_cross('int1e_ovlp',mol,small)@t;x=solve(sd,cross)-core@(core.T@cross)
  def power(a,p):v,u=eigh(a);assert v.min()>1e-10;return (u*v**p)@u.T
  lift=x@power(x.T@sd@x,-.5)@power(s.overlap,.5);assert np.max(abs(lift.T@sd@lift-s.overlap))<1e-8
  np.savez_compressed(ep,embedding=lift)
 with np.load(ep) as z:lift=z['embedding'].copy()
 if not (folder/'projected_eri.npz').exists():np.savez_compressed(folder/'projected_eri.npz',eri=ao2mo.kernel(mol,(lift,)*4,compact=False).reshape((n,)*4))
 # Complete symmetric AO chart; only this chart, not its zero placeholder,
 # enters the final CAM calibration. No old empirical response is required.
 if not (folder/'model.npz').exists():
  ops=[]
  for i in range(n):
   for j in range(i+1):
    a=np.zeros((n,n));a[i,j]=a[j,i]=1. if i==j else 2**-.5;ops.append(a)
  ops=np.array(ops);nf=len(ops);np.savez_compressed(folder/'model.npz',operators=ops,linear=np.zeros(nf),charge=np.zeros((nf,nf)),spin=np.zeros((nf,nf)),q0=np.zeros(nf))
 link(folder/'model.npz',src/'qualified_spectral_response/anchor/model.npz')
 transition=src/('native_transition_'+identity+'.npz')
 if not transition.exists():
  target=np.zeros((no,n-no))
  if target_model is not None:
   from matrix_gfn2.nosi_surface import PreparedNOSIModel
   original=PreparedNOSIModel(target_model);value=original.evaluate(calc,xyz,properties=('energy',));index=original.state_ids.index(identity);c=value.state_coefficients[index];target=np.outer(s.coefficients[:,:no].T@s.overlap@c[:,0],s.coefficients[:,no:].T@s.overlap@c[:,no]);target/=np.linalg.norm(target)
  else:
   small,t=source_molecule(s);origin=s.coordinates_bohr.mean(0);_,_,axes=np.linalg.svd(s.coordinates_bohr-origin,full_matrices=True);normal=axes[-1];reflection=np.eye(3)-2*np.outer(normal,normal);other=small.copy();other.set_geom_((s.coordinates_bohr-origin)@reflection.T+origin,unit='Bohr');rotation=np.eye(small.nao_nr());loc=small.ao_loc_nr()
   for shell in range(small.nbas):
    if small.bas_angular(shell)==1:rotation[loc[shell]:loc[shell+1],loc[shell]:loc[shell+1]]=reflection
   op=t.T@gto.intor_cross('int1e_ovlp',small,other)@rotation@t;characters=np.diag(s.coefficients.T@op@s.coefficients)
   hsign=1 if name.startswith('Formaldehyde') else -1;holes=np.flatnonzero(characters[:no]*hsign>.99);particles=np.flatnonzero(characters[no:]<-.99);assert len(holes) and len(particles),(name,characters);target[holes[-1],particles[0]]=1.
  np.savez_compressed(transition,amplitude=target)
  write(src/('native_assignment_'+identity+'.json'),dict(rule='fixed valence orbital character: highest occupied n or pi to lowest virtual pi; old-model NTO transport for re-anchoring',target_model=None if target_model is None else str(target_model),amplitude=target,energy_labels_read=False,TDA_required=False))
 manifest=dest/'manifest.json';rows=json.loads(manifest.read_text())['states'] if manifest.exists() else []
 row=dict(id=identity,molecule=name,ordinal_same_symmetry=1,symmetry='HOMO-LUMO assigned',family='unscored')
 if not any(r['id']==identity and r['molecule']==name for r in rows):rows.append(row);write(manifest,dict(states=rows,target_energy_labels_included=False))
 # New cohort inputs are shared with the established basis construction.
 link(dest,Q/'data'/cohort)
 return row

def run(name,identity,xyzfile,cohort='closing_blind',target_model=None):
 row=initial(name,identity,xyzfile,cohort,target_model)
 anchor(cohort,name);full_reference(cohort,name,prepare_only=True);calibrate(cohort,name)
 # Native preparation needs the full reference and F response; its own F/Q
 # one-body average preserves the native orbitals exactly.
 from native_batch import install_native_batch
 install_native_batch(prepare(cohort,name),CAM/cohort/name)
 engine=module('closing_native_engine',NATIVE/'run.py');engine.prepare_native(cohort,name)
 import heldout_predict as hp,nosi_coherent as nc
 hp.DEST=NATIVE;nc.DEST=NATIVE;nc.HERE=NATIVE/'nosi_coherent'
 def native_transition(case,row):
  with np.load(case['src']/('native_transition_'+row['id']+'.npz')) as z:q=z['amplitude'].copy()
  seed=case['src']/('native_seed_'+row['id']+'.npz')
  if seed.exists():case['retained_seed']=seed
  s=case['s'];no=case['no'];return s.coefficients[:,:no]@q@s.coefficients[:,no:].T
 # Previously completed Formaldehyde is retained; new explicit valence seeds
 # remove the need for a diffuse DFT-TDA projection in a minimal basis.
 hp.target_transition=native_transition
 hp.predict_molecule(cohort,name);nc.run(cohort,name)
 sys.path.insert(0,str(Q));sys.path.insert(0,str(REPAIR))
 import expanded_cid_fast as ex,even_sector as ev,export_screened_model as export
 # The same canonical algorithm and fixed active space used for the exposed set.
 item=dict(cohort=cohort,molecule=name,states=[identity]);ex.run(item,5);ev.run(item);export.export(item)
 write(HERE/'predictions'/cohort/(name+'.json'),dict(item=item,result=json.loads((REPAIR/'even_sector_5'/cohort/name/'result.json').read_text()),model_sha256=sha(REPAIR/'prepared_models'/cohort/name/'model.json'),protocol_sha256=sha(HERE/'protocol.json'),scoring_labels_read=False))
 print(name,'fixed candidate completed',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('identity');p.add_argument('xyz');p.add_argument('--cohort',default='closing_blind');a=p.parse_args()
 with threadpool_limits(2):run(a.name,a.identity,a.xyz,a.cohort)
