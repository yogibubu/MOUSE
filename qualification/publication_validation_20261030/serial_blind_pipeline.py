"""Serial scheduling repair; frozen scientific workers and all checkpoints unchanged."""
from pathlib import Path
import socket,os,sys,json,signal,time,datetime,subprocess,hashlib,fcntl,argparse
assert socket.getfqdn()=='oracle.sns.it'
P=Path(__file__).resolve().parent
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
def read(p):return json.loads(p.read_text())
def write(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def alive(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
 except FileNotFoundError:return False
def progress(stage,name,**kw):
 v=dict(host=socket.getfqdn(),stage=stage,molecule=name,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),numerical_labels_read=False,**kw)
 write(P/'blind/pipeline_progress.json',v);print(json.dumps(v),flush=True)
def adopt(parent,waiter):
 cmd=Path(f'/proc/{parent}/cmdline').read_bytes()
 assert b'blind_reference.py' in cmd and b'--worker' not in cmd
 cmd=Path(f'/proc/{waiter}/cmdline').read_bytes()
 assert b'blind_mouse.py' in cmd and b'--worker' not in cmd
 # Suspend only the orchestration parent, never its active electronic worker.
 os.kill(parent,signal.SIGSTOP)
 old=read(P/'blind/references/progress.json');pid=old['pid'];name=old['molecule']
 os.kill(waiter,signal.SIGTERM) # waiter has no worker; serial lock is still held.
 deadline=datetime.datetime.fromisoformat(old['started_utc']).timestamp()+3600
 progress('adopting_existing_reference',name,pid=pid,preserved_checkpoint=True)
 while alive(pid):
  if time.time()>=deadline:
   os.killpg(pid,signal.SIGKILL)
   write(P/'blind/references'/name/'failure.json',dict(status='timeout',returncode=-9,host=socket.getfqdn(),checkpoint_preserved=True,scope='original unchanged 3600-second budget'))
   break
  time.sleep(3)
 os.kill(parent,signal.SIGKILL) # worker finished; release old batch-wide lock.
 progress('adoption_complete',name,reference_result_exists=(P/'blind/references'/name/'result.json').exists())
def launch(kind,row):
 name=row['molecule'];d=P/'blind'/('references' if kind=='reference' else 'mouse')/name;d.mkdir(exist_ok=True)
 source=P/('blind_reference.py' if kind=='reference' else 'blind_mouse.py')
 arg=row['id'] if kind=='reference' else name
 with (d/'pipeline_worker.log').open('a') as log:
  proc=subprocess.Popen(['/usr/bin/time','-v','-o',str(d/'pipeline_resources.txt'),sys.executable,str(source),'--worker',arg],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=os.environ)
  state=dict(status='running',molecule=name,pid=proc.pid,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),serial_pipeline=True,numerical_labels_read=False)
  write(d.parent/'progress.json',state);progress(kind,name,pid=proc.pid)
  try:rc=proc.wait(timeout=3600)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();rc=-9
 if rc:write(d/'failure.json',dict(status='timeout' if rc==-9 else 'execution_error',returncode=rc,checkpoint_preserved=True,serial_pipeline=True))
 return rc
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--adopt-reference-parent',type=int);ap.add_argument('--replace-mouse-waiter',type=int);a=ap.parse_args()
 singleton=(P/'blind/pipeline.lock').open('a');fcntl.flock(singleton,fcntl.LOCK_EX|fcntl.LOCK_NB)
 import blind_mouse as mouse
 assert read(P/'blind/mouse/protocol.json')==mouse.get_plan()
 plan=dict(host=socket.getfqdn(),threads=2,concurrent_electronic_workers=1,policy='consume existing references first; then each new reference immediately followed by MOUSE; no completed or partial reference is rerun; previous reference timeouts retained for targeted recovery',source_sha256={n:sha(P/n) for n in ['blind_reference.py','blind_mouse.py','serial_blind_pipeline.py']},manifest_sha256=sha(P/'blind/manifest.json'),model_or_assignment_changes=False,numerical_labels_read=False)
 planpath=P/'blind/pipeline_protocol.json'
 if planpath.exists():assert read(planpath)==plan
 else:write(planpath,plan)
 if a.adopt_reference_parent:adopt(a.adopt_reference_parent,a.replace_mouse_waiter)
 lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 rows={}
 for r in read(P/'blind/manifest.json')['states']:rows.setdefault(r['molecule'],r)
 ordered=sorted(rows.values(),key=lambda r:(r['n_atoms'],r['molecule']))
 # First deliver OO predictions for already available complete/partial spectra.
 for r in ordered:
  if (P/'blind/references'/r['molecule']/'result.json').exists() and not (P/'blind/mouse'/r['molecule']/'result.json').exists():launch('mouse',r)
 # Continue unfinished references; consume each immediately when it finishes.
 for r in ordered:
  d=P/'blind/references'/r['molecule']
  if (d/'result.json').exists() or (d/'failure.json').exists():continue
  launch('reference',r)
  if (d/'result.json').exists() and not (P/'blind/mouse'/r['molecule']/'result.json').exists():launch('mouse',r)
 progress('pass_complete',None,scoring='not performed; all-state ledger audit required')
