"""Experimental energy-only zero-frequency subtraction on saved pigment pencils."""
from run_pigment_cid import *
from matrix_gfn2._nosi_static_subtraction import static_subtraction
from matrix_gfn2._nosi_selected_roots import solve_full
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--frontier',type=int,required=True);p.add_argument('--screening-frontier',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--roots',type=int,default=8);p.add_argument('--threads',type=int,default=4);p.add_argument('--counterterm');p.add_argument('--reuse-diagnostic');p.add_argument('--eigensolver',choices=['arpack','lobpcg','davidson'],default='arpack');a=p.parse_args()
folder=HERE/f'pigment_cid_transition/{a.name}/screening_{a.screening_frontier}/frontier_{a.frontier}'
if (HERE/a.output).exists():
 print('REUSE completed diagnostic',a.output,flush=True);raise SystemExit(0)
with threadpool_limits(4):
 start=time.monotonic();report=json.loads((folder/'result.json').read_text());nbase=1+2*len(report['retained_ROKS_ordinals'])
 digest=hashlib.sha256()
 with (folder/'matrices.npz').open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
 identity=digest.hexdigest();checkpoint=HERE/a.counterterm if a.counterterm else HERE/a.output.replace('.json','_counterterm.npz')
 with np.load(folder/'matrices.npz') as z:h=z['h'];s=z['s'];eg=float(z['ground_energy'])
 projection=s[0,1:]/s[0,0];sp=s[1:,1:]-np.outer(s[1:,0],projection)
 hp=h[1:,1:]-np.outer(h[1:,0],projection)-np.outer(projection,h[0,1:])+h[0,0]*np.outer(projection,projection)
 del h,s
 hp-=eg*sp
 print('PRIMARY',nbase-1,'COMPLEMENT',len(hp)-nbase+1,flush=True)
 if checkpoint.exists():
  with np.load(checkpoint) as z:
   if 'source_sha256' not in z or str(z['source_sha256'])!=identity:raise ValueError('counterterm checkpoint source identity mismatch')
   r=z['r'].copy();delta=z['delta'].copy();certificate=json.loads(str(z['certificate']))
  print('REUSE certified counterterm',flush=True)
 elif a.reuse_diagnostic:
  legacy_path=HERE/a.reuse_diagnostic;legacy=json.loads(legacy_path.read_text())
  assert (legacy['molecule'],legacy['frontier'],legacy['screening_frontier'])==(a.name,a.frontier,a.screening_frontier)
  receipts=[record for ledger in HERE.glob('*_evidence_20261102/large_artifacts.json') for record in json.loads(ledger.read_text()) if record['path']==str(folder/'matrices.npz') and record['sha256']==identity]
  if not receipts:raise ValueError('legacy counterterm requires independently archived source matrix identity')
  with np.load(legacy_path.with_suffix('.npz')) as z:delta=z['primary_counterterm'].copy();old_ci=z['ci'].copy();old_e=z['energies'].copy()
  primary=nbase-1;x=np.linalg.solve(sp[:primary,:primary],sp[:primary,primary:]);r=np.vstack((np.eye(primary),x.T))
  residual=hp@old_ci+(r@delta)@(r.T@old_ci)-(sp@old_ci)*old_e
  error=float(np.max(np.linalg.norm(residual,axis=0)))
  if error>1e-8:raise ValueError(('legacy subtracted eigenpairs do not match current source',error))
  certificate=dict(legacy['subtraction_certificate'],reused_from=str(legacy_path),legacy_eigenpair_replay_residual_hartree=error)
  np.savez_compressed(checkpoint,r=r,delta=delta,certificate=json.dumps(certificate),source_sha256=identity)
  print('REUSE legacy counterterm with archived source identity and full eigenpair replay',flush=True)
 else:
  r,delta,certificate=static_subtraction(hp,sp,np.arange(nbase-1))
  np.savez_compressed(checkpoint,r=r,delta=delta,certificate=json.dumps(certificate),source_sha256=identity)
 print('SUBTRACTION',certificate,flush=True)
 hp+=(r@delta)@r.T
 with np.load(folder/'eigenstates.npz') as z:guess=z['ci'][1:].copy()
 if a.eigensolver!='davidson':guess=guess[:,:a.roots]
 elif guess.shape[1]<a.roots:guess=np.column_stack((guess,np.random.default_rng(715).normal(size=(len(guess),a.roots-guess.shape[1]+8))))
 e,c,eigen_certificate=solve_full(hp,sp,a.roots,energy_shift=0.,algorithm=a.eigensolver,initial=guess,block_workers=a.threads)
 result=dict(molecule=a.name,frontier=a.frontier,screening_frontier=a.screening_frontier,source_matrices_sha256=identity,states=[dict(root=i,excitation_ev=float(value*27.211386245988)) for i,value in enumerate(e,1)],subtraction_certificate=certificate,eigensolver_certificate=eigen_certificate,wall_seconds=time.monotonic()-start,scope='Experimental energy-only static-subtraction diagnostic; no oscillator-strength or nuclear-derivative qualification; not promoted to production')
 np.savez_compressed(HERE/a.output.replace('.json','.npz'),energies=e,ci=c,primary_counterterm=delta);write(HERE/a.output,result);print(json.dumps(result,indent=2),flush=True)
