"""Label-free PBE0 reference preparation for the frozen blind manifest.
Only public manifest and its XYZs are read; sealed labels and QUEST metadata
are never opened. Reference TD spectra assist identity, not prediction scoring.
"""
from pathlib import Path
import socket,os,json,hashlib,argparse,time,traceback,subprocess,signal,datetime
assert socket.getfqdn()=='oracle.sns.it', 'Remote electronic jobs only'
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
P=Path(__file__).resolve().parent;ROOT=P/'blind/references';ROOT.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def plan():return dict(schema='mouse.blind.reference.v1',host=socket.getfqdn(),manifest_sha256=sha(P/'blind/manifest.json'),source_sha256=sha(Path(__file__)),xc='PBE0',basis='aug-cc-pVDZ',grid_level=3,threads=2,scf_tolerance=1e-10,scf_max_cycles=150,newton_fallback=True,tda_roots=40,tda_tolerance=1e-7,tda_max_cycles=150,timeout_seconds_per_molecule=3600,max_memory_mb=24000,read_inputs=['blind/manifest.json','blind/geometries/*.xyz'],sealed_labels_read=False,root_assignment='deferred to frozen symmetry, ordinal, character and compact-projection checks; no TBE proximity',window_policy='record coverage of 20 eV; no claim of complete spectrum when 40 roots do not reach window')
def worker(sid):
 import numpy as np
 from pyscf import gto,dft,lib,symm,scf
 from pyscf.tdscf import rhf
 lib.num_threads(2)
 manifest=json.loads((P/'blind/manifest.json').read_text());row=next(x for x in manifest['states'] if x['id']==sid)
 d=ROOT/row['molecule'];d.mkdir(exist_ok=True);xyz=P/'blind/geometries'/f'{sid}.xyz'
 assert sha(xyz)==row['geometry_sha256']
 rule=plan();request={'plan_sha256':sha(ROOT/'protocol.json'),'geometry_sha256':sha(xyz)}
 if (d/'result.json').exists():
  old=json.loads((d/'result.json').read_text());assert old['request']==request
  if old['status']=='reference_ready':return
 lines=xyz.read_text().splitlines();n=int(lines[0]);atoms='\n'.join(lines[2:2+n]);start=time.perf_counter()
 mol=gto.M(atom=atoms,unit='Angstrom',basis='aug-cc-pVDZ',charge=0,spin=0,symmetry=True,verbose=4,output=str(d/'pyscf.log'),max_memory=24000)
 mf=dft.RKS(mol);mf.xc='PBE0';mf.grids.level=3;mf.small_rho_cutoff=0;mf.conv_tol=1e-10;mf.max_cycle=150;mf.chkfile=str(d/'backend.chk')
 ground=d/'reference.npz';scf_start=time.perf_counter()
 if ground.exists():
  with np.load(ground) as f:
   assert str(f['request_hash'])==hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
   mf.mo_coeff=f['coefficients'];mf.mo_occ=f['occupations'];mf.mo_energy=f['orbital_energies'];mf.e_tot=float(f['energy_hartree']);mf.converged=True
 else:
  dm=None
  if (d/'backend.chk').exists():
   try:dm=mf.init_guess_by_chkfile(str(d/'backend.chk'))
   except Exception:pass
  mf.kernel(dm0=dm)
  if not mf.converged:
   base=mf;mf=mf.newton();mf.max_cycle=100;mf.kernel(base.mo_coeff,base.mo_occ)
  if not mf.converged:raise RuntimeError('DFT reference SCF did not converge')
  temp=d/'reference.tmp.npz';np.savez_compressed(temp,coefficients=mf.mo_coeff,occupations=mf.mo_occ,orbital_energies=mf.mo_energy,overlap=mf.get_ovlp(),energy_hartree=mf.e_tot,atomic_numbers=mol.atom_charges(),coordinates_bohr=mol.atom_coords(),request_hash=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest());temp.replace(ground)
  write(d/'ground_result.json',dict(request=request,host=socket.getfqdn(),seconds=time.perf_counter()-scf_start,nao=mol.nao_nr(),converged=True,point_group=mol.groupname))
 td=mf.TDA();nocc=int(np.count_nonzero(mf.mo_occ==2));nv=int(np.count_nonzero(mf.mo_occ==0));td.nstates=min(40,nocc*nv);td.conv_tol=1e-7;td.max_cycle=150;td.max_memory=24000
 stage=time.perf_counter();td.kernel()
 amps=np.array([x for x,y in td.xy]);energies=np.asarray(td.e);converged=np.asarray(td.converged,dtype=bool)
 orbsym=scf.hf_symm.get_orbsym(mol,mf.mo_coeff);xsym=symm.direct_prod(orbsym[mf.mo_occ==2],orbsym[mf.mo_occ==0],mol.groupname)
 labels=[rhf._analyze_wfnsym(td,xsym,x) for x in amps]
 temp=d/'backend_tda.tmp.npz';np.savez_compressed(temp,amplitudes=amps.reshape(len(amps),-1),energies_hartree=energies,converged=converged,orbital_symmetries=orbsym,state_symmetries=np.array(labels),request_hash=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest());temp.replace(d/'backend_tda.npz')
 ev=energies*27.211386245988
 result=dict(status='reference_ready' if np.all(converged) else 'reference_partial',request=request,host=socket.getfqdn(),molecule=row['molecule'],manifest_states=[x['id'] for x in manifest['states'] if x['molecule']==row['molecule']],nao=mol.nao_nr(),nocc=nocc,nvirtual=nv,point_group=mol.groupname,tda_seconds=time.perf_counter()-stage,seconds_current_run=time.perf_counter()-start,roots=len(ev),converged_roots=int(sum(converged)),covers_20_ev=bool(len(ev) and ev[-1]>=20),states=[dict(root=i+1,symmetry=str(l),energy_ev=float(e),converged=bool(c)) for i,(e,l,c) in enumerate(zip(ev,labels,converged))],reference_labels_read=False)
 write(d/'result.json',result);print(row['molecule'],result['status'],len(ev),flush=True)
def orchestrate():
 import fcntl
 lock=(ROOT/'compute.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 rule=plan();f=ROOT/'protocol.json'
 if f.exists():assert json.loads(f.read_text())==rule
 else:write(f,rule)
 manifest=json.loads((P/'blind/manifest.json').read_text());assert sha(P/'blind/manifest.json')==(P/'blind/manifest.sha256').read_text().strip()
 molecules={}
 for r in manifest['states']:molecules.setdefault(r['molecule'],r)
 ordered=sorted(molecules.values(),key=lambda x:(x['n_atoms'],x['molecule']))
 for r in ordered:
  d=ROOT/r['molecule'];d.mkdir(exist_ok=True)
  if (d/'result.json').exists() and json.loads((d/'result.json').read_text()).get('status')=='reference_ready':continue
  with (d/'worker.log').open('a') as log:
   proc=subprocess.Popen(['/usr/bin/time','-v','-o',str(d/'resources.txt'),os.sys.executable,str(Path(__file__)),'--worker',r['id']],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=os.environ)
   write(ROOT/'progress.json',dict(status='running',molecule=r['molecule'],pid=proc.pid,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sealed_labels_read=False))
   try:rc=proc.wait(timeout=3600)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();rc=-9
  if rc:
   write(d/'failure.json',dict(status='timeout' if rc==-9 else 'execution_error',returncode=rc,host=socket.getfqdn(),checkpoint_preserved=True));print(r['molecule'],'failed',rc,flush=True)
 write(ROOT/'progress.json',dict(status='reference_pass_complete',sealed_labels_read=False,results=[dict(molecule=r['molecule'],result_exists=(ROOT/r['molecule']/'result.json').exists(),failure_exists=(ROOT/r['molecule']/'failure.json').exists()) for r in ordered]))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--worker');args=a.parse_args()
 if args.worker:worker(args.worker)
 else:orchestrate()
