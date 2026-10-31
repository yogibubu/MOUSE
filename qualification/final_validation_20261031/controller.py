"""Serial resumable four-gate campaign. No labels are unsealed by this runner."""
from pathlib import Path
import socket,sys,subprocess,json,hashlib,fcntl,os,signal,time,datetime
assert socket.getfqdn()=='oracle.sns.it'
R=Path(__file__).resolve().parent
if '--wait-pid' in sys.argv:
 pid=int(sys.argv[sys.argv.index('--wait-pid')+1]);begin=time.monotonic()
 while time.monotonic()-begin<7200:
  try:os.kill(pid,0)
  except ProcessLookupError:break
  time.sleep(2)
 else:raise RuntimeError('Prior child is still running; do not duplicate it')
lock=(R/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def write(p,d):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sources={p.name:sha(p) for p in R.glob('*.py')};protocol_sha=sha(R/'protocol.json')
start=R/'launch_manifest_v2.json'
manifest=dict(host=socket.getfqdn(),protocol_sha256=protocol_sha,source_sha256=sources,independent_manifest_sha256=sha(R/'independent/manifest.json'),started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),shared_lock='parent process dispatches children serially; each child holds campaign compute.lock',sealed_labels_read=False)
if not start.exists():write(start,manifest)
else:assert json.loads(start.read_text())['source_sha256']==sources,'Sources changed; archive old launch and explicitly version the revised runner'
tasks=[]
def add(tag,args,complete):tasks.append(dict(tag=tag,args=args,complete=complete))
for n,s in [('Furan','B022'),('Cyclopentadiene','B016')]:
 for scale in ['0.98','1.02']:add(f'anchor_{n}_{scale}',['anchor_energy.py',n,s,scale],f'anchor_energy/{n}/{scale}/result.json')
add('independent_Ammonia',['independent.py','Ammonia'],'independent/predictions/Ammonia/I006/result.json')
add('geometry_Furan',['geometry.py','Furan','B022'],'geometry/Furan/result.json')
for n in ['nitroxyl','acetaldehyde','acetone','Furan','Cyclopentadiene','Phthalimide']:
 for prop in ['tda','energy','gradient','hessian']:
  for repeat in range(3):add(f'cost_{n}_{prop}_{repeat}',['cost.py',n,prop,str(repeat)],f'cost/{n}/{prop}/{repeat}/result.json')
for n in ['Criegee_Intermediate','Cyclobutadiene','Hexatriene','Benzoxadiazole']:add(f'independent_{n}',['independent.py',n],f'independent/predictions/{n}/pass_complete.json')
add('geometry_Cyclopentadiene',['geometry.py','Cyclopentadiene','B016'],'geometry/Cyclopentadiene/result.json')
for scale in ['0.98','1.02']:add(f'anchor_Phthalimide_{scale}',['anchor_energy.py','Phthalimide','B024',scale],f'anchor_energy/Phthalimide/{scale}/result.json')
# Complete an inexpensive cost case early, then independent science before
# the repeated largest-system Hessian timings. Numerical protocols unchanged.
def priority(task):
 tag=task['tag']
 if tag.startswith('anchor_') and 'Phthalimide' not in tag:return 0
 if tag=='independent_Ammonia':return 1
 if tag=='geometry_Furan':return 2
 if tag.startswith('cost_nitroxyl_'):return 3
 if tag.startswith('independent_'):return 4
 if tag.startswith('geometry_'):return 5
 if tag.startswith('anchor_'):return 6
 return 7
tasks.sort(key=priority)
ledger=R/'task_ledger.json';history=json.loads(ledger.read_text()) if ledger.exists() else []
for task in tasks:
 if (R/task['complete']).exists():continue
 tag=task['tag'];folder=R/'tasks'/tag;folder.mkdir(parents=True,exist_ok=True)
 if (folder/'exit.json').exists():continue # bounded attempts; explicit checkpointed recovery, never a silent rerun
 write(R/'progress.json',dict(status='running',task=tag,completed_outputs=sum((R/t['complete']).exists() for t in tasks),total_tasks=len(tasks),host=socket.getfqdn(),updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
 with (folder/'stdout.log').open('a') as log:
  proc=subprocess.Popen(['/usr/bin/time','-v','-o',str(folder/'resources.txt'),sys.executable,*task['args']],cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  write(folder/'pid.json',dict(pid=proc.pid,args=task['args']));begin=time.monotonic()
  try:code=proc.wait(timeout=7200)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();code=-9
 row=dict(tag=tag,exit_code=code,seconds=time.monotonic()-begin,output_exists=(R/task['complete']).exists(),checkpoints_preserved=True)
 write(folder/'exit.json',row);history.append(row);write(ledger,history)
write(R/'progress.json',dict(status='attempt_pass_complete',all_tasks_attempted=True,completed_outputs=sum((R/t['complete']).exists() for t in tasks),total_tasks=len(tasks),failed_tasks=[r['tag'] for r in history if r['exit_code']],sealed_labels_read=False))
