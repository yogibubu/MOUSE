"""Close the retained 14-state campaign using assigned endpoint searches."""
from pathlib import Path
import os,socket,sys,json,hashlib,subprocess,datetime,shutil,fcntl,signal
assert socket.getfqdn()=='oracle.sns.it'
R=Path('/home/enzo/MOUSE-state-following-20261031')
V=Path('/home/enzo/MOUSE-orbital-performance-20261031')
P=Path('/home/enzo/MOUSE-publication-validation-20261030');D=R/'reduced_validation'
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def progress(stage,**kw):write(V/'progress.json',dict(host=socket.getfqdn(),stage=stage,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw))
scope=read(R/'reduced_scope.json')['primary_states'];manifest=read(P/'blind/manifest.json')['states'];scope=scope+[next(x for x in manifest if x['id']=='B039')]
def ledger():
 rows=[dict(x,result=read(V/'results'/x['molecule']/x['id']/'result.json') if (V/'results'/x['molecule']/x['id']/'result.json').exists() else dict(status='pending')) for x in scope]
 closed=all(x['result']['status']=='matched' for x in rows)
 write(V/'ledger.json',dict(host=socket.getfqdn(),expected_states=14,recorded_states=sum(x['result']['status']!='pending' for x in rows),matched_states=sum(x['result']['status']=='matched' for x in rows),closed=closed,rows=rows,accuracy_scoring_performed=False))
 byid={x['id']:x['result']['status'] for x in rows}
 write(V/'scope_ledger.json',dict(original_results_preserved=True,rows=[dict(id=x['id'],molecule=x['molecule'],status=byid.get(x['id'],'deferred_user_scope_reduction')) for x in manifest]))
 return closed
def launch(stage,name,args,timeout=1800):
 folder=V/'runs'/name/stage;folder.mkdir(parents=True,exist_ok=True)
 progress(stage,molecule=name,status='running')
 compute=None
 if Path(args[0]).name not in ['repair_assigned_parent.py','assigned_pair_worker.py']:
  compute=(P/'blind/references/compute.lock').open('a');fcntl.flock(compute,fcntl.LOCK_EX|fcntl.LOCK_NB)
 with (folder/'worker.log').open('a') as f:
  proc=subprocess.Popen(['/usr/bin/time','-v','-o',str(folder/'resources.txt'),sys.executable]+list(map(str,args)),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=proc.wait(timeout=timeout)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGTERM);proc.wait();rc=124
 if compute:compute.close()
 if rc:write(folder/'failure.json',dict(returncode=rc,checkpoints_preserved=True))
 return rc==0
def main():
 # Child electronic workers lock independently; the orchestration lock prevents
 # two copies of this scheduler without holding the shared compute lock.
 lock=(V/'orchestrator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 source=[V/n for n in ['assigned_pipeline.py','repair_assigned_parent.py','assigned_pair_worker.py','assigned_pair_audit.py','assigned_tda.py','assigned_score.py','fast_prepare.py']]
 source+=sorted((V/'source/packages/matrix-gfn2/src/matrix_gfn2').glob('*.py'))
 source+=[R/n for n in ['checkpointed_reference.py','reference_eigensolver.py','reduced_reference.py','reduced_spectral_diagnostic.py','reduced_projected_model.py']]
 plan=dict(schema='mouse.assigned.endpoint.repair.v3',source_sha256={str(p):sha(p) for p in source},states=[x['id'] for x in scope],scope_sha256=sha(R/'reduced_scope.json'),parent_campaign_protocol_sha256=sha(D/'protocol.json'),weights=[.5,.5],lambda_exchange=.9,orbital_gap_scale=1.,energy_offset_ev=0.,molecules=10,threads=2,reference_energy_labels_read=False,model_rule='unchanged canonical-gap molecular selection and full AO-pair charge/spin kernels',initialization='retained qualified full parent, assigned NTO and native seed; Newton across seeds before automatic-index freeze/release GMF; no F-to-Q branch requirement',identity='overlap with unit-normalized fixed projected target divided by its leading singular value, squared >0.5; this is an attainable single-pair fidelity ratio, not a many-electron state probability; each occupied/virtual sector norm >=0.7; local CSF step >=0.9; raw projected-overlap margin >=0.05 against every available converged same-symmetry DFT root; no extra global active-orbital overlap restriction',identity_amendment='replaces rank-one fidelity >=0.90 and redundant global active-pair anchor; changes assignment criteria, not functional, convergence tolerances or reference energies; applied to all 14 states; raw-majority prototype rejected before remaining six predictions because it also penalizes compact-space single-pair representability',stationarity='parent residual <=1e-9; actual analytic saddle index; coupled residual <=1e-8 and minimum absolute curvature >1e-7; all three restricted grounds index zero with positive curvature',coverage='available-root audit only, not proof of complete spectrum',accuracy='all labels remain sealed until 14 OO assignments and 10 finalized TDA files; separately report 6 initial development, 2 nitropyridine solver repair and 6 remaining states; no post-score fitting')
 if (V/'protocol.json').exists():assert read(V/'protocol.json')==plan,'frozen sources changed'
 else:write(V/'protocol.json',plan)
 for row in read(D/'ledger.json')['rows']:
  if row['result']['status']=='matched':
   p=V/'results'/row['molecule']/row['id']/'result.json'
   if not p.exists():write(p,row['result'])
 tda=V/'tda'
 if not tda.exists():tda.symlink_to(D/'tda',target_is_directory=True)
 ledger();unresolved=[]
 for name in ['Nitropyridine_N-oxide','Adenine','Phthalimide','Quinoxaline']:
  selected=[x for x in scope if x['molecule']==name]
  if not (P/'blind/references'/name/'result.json').exists():
   if not launch('reference',name,[R/'checkpointed_reference.py',name]):unresolved.append(name+':reference');continue
  if not (D/'models'/name/'model.npz').exists():
   okay=launch('prepare',name,[V/'fast_prepare.py',name])
   if not okay:
    log=(V/'runs'/name/'prepare/worker.log').read_text()
    if 'spectral transfer loses the native occupied/virtual gap' not in log:unresolved.append(name+':prepare');continue
    if not launch('ordering',name,[R/'reduced_spectral_diagnostic.py',name]):unresolved.append(name+':ordering');continue
   if not launch('selection',name,[R/'reduced_reference.py',name]):unresolved.append(name+':selection');continue
  for row in selected:
   sid=row['id'];out=V/'results'/name/sid/'result.json'
   if out.exists() and read(out)['status']=='matched':continue
   parents=True
   for profile in ['full','spectral']:
    launch(sid+'_'+profile,name,[V/'repair_assigned_parent.py',name,sid,profile,'auto','majority'],timeout=240)
    report=V/'assigned_parents'/name/sid/profile/'majority_v1/result.json'
    if not report.exists() or not read(report).get('qualified'):parents=False
   if not parents:unresolved.append(sid+':assigned_parent');continue
   if out.exists():out.rename(out.with_name('result_before_resume.json'))
   launch(sid,name,[V/'assigned_pair_worker.py',name,sid],timeout=240)
   launch('audit',name,[V/'assigned_pair_audit.py'],timeout=120)
   ledger()
   if not out.exists() or read(out)['status']!='matched':unresolved.append(sid+':joint_or_assignment')
  launch('tda',name,[V/'assigned_tda.py',name],timeout=240)
  ledger()
 if not ledger():progress('repair_required',status='needs_attention',unresolved=unresolved,checkpoints_preserved=True);return
 names=sorted({x['molecule'] for x in scope})
 if not all((V/'tda'/n/'result.json').exists() for n in names):progress('tda_required',status='needs_attention');return
 files=[V/'results'/x['molecule']/x['id']/'result.json' for x in scope]+[V/'tda'/n/'result.json' for n in names]
 write(V/'prediction_lock.json',dict(protocol_sha256=sha(V/'protocol.json'),files_sha256={str(p):sha(p) for p in files}))
 okay=launch('scoring','all',[V/'assigned_score.py'],timeout=60)
 progress('complete' if okay else 'scoring_requires_inspection',status='complete' if okay else 'needs_attention',matched_states=14,molecules=10)
if __name__=='__main__':
 try:main()
 except Exception as exc:
  import traceback
  progress('orchestration_exception',status='needs_attention',reason=str(exc),checkpoints_preserved=True);print(traceback.format_exc(),flush=True)
