"""Finite rerun of affected properties after exact analytic contraction changes."""
from pathlib import Path
import socket,subprocess,json,time,sys,hashlib,datetime,shutil
assert socket.getfqdn()=='oracle.sns.it'
r=Path(__file__).resolve().parent;deadline=time.monotonic()+1800
while not (r/'hessian_acceleration/qualification/result.json').exists():
 if time.monotonic()>deadline:raise TimeoutError('Qualification has not completed')
 time.sleep(3)
assert json.loads((r/'hessian_acceleration/qualification/result.json').read_text())['status']=='pass'
old=r/'cost_before_hessian_acceleration';old.mkdir(exist_ok=True)
protocol=r/'hessian_acceleration_benchmark_amendment.json'
if not protocol.exists():
 protocol.write_text(json.dumps(dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reason='User requests a completed Phthalimide analytic Hessian. Exact early feature contraction, batched response and reuse of the independently assembled orbital/CI Hessian change only the evaluation order.',properties_remeasured=['gradient','hessian'],unchanged_properties_reused=['energy','tda'],weights=[.5,.5],lambda_exchange=.9,energy_offset_ev=0,repetitions='three small; one large; previous prespecified omissions unchanged',old_results_retained=True,qualification_sha256=hashlib.sha256((r/'hessian_acceleration/qualification/result.json').read_bytes()).hexdigest()),indent=2)+'\n')
 for name in ['nitroxyl','acetaldehyde','acetone','Furan','Cyclopentadiene','Phthalimide']:
  for prop in ['gradient','hessian']:
   dest=old/name/prop;dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();shutil.move(str(r/'cost'/name/prop),str(dest))
   src=r/'cost_executions'/name/prop
   if src.exists():
    d=old/'executions'/name/prop;d.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(src),str(d))
for name in ['nitroxyl','acetaldehyde','acetone','Furan','Cyclopentadiene','Phthalimide']:
 for prop in ['gradient','hessian']:
  for repeat in range(3):
   folder=r/'hessian_acceleration/benchmark'/name/prop/str(repeat);folder.mkdir(parents=True,exist_ok=True)
   with (folder/'stdout.log').open('a') as f:
    subprocess.run(['/usr/bin/time','-v','-o',str(folder/'resources.txt'),sys.executable,str(r/'cost.py'),name,prop,str(repeat)],cwd=r,stdout=f,stderr=subprocess.STDOUT,check=True)
   print(name,prop,repeat,flush=True)
rows=[json.loads(p.read_text()) for p in (r/'cost').glob('*/*/*/result.json')]
assert len(rows)==72
assert all(p['status']=='complete' for p in rows if p['property']=='hessian' and p['repeat']==0)
(r/'hessian_acceleration_benchmark_complete.json').write_text(json.dumps(dict(status='pass',complete_measurements=sum(p['status']=='complete' for p in rows),omitted_repeats=sum(p['status']=='not_repeated_due_to_prespecified_budget' for p in rows),all_six_hessians_complete=True),indent=2)+'\n')
subprocess.run([sys.executable,str(r/'collect_final.py')],check=True)
print('ALL ANALYTIC HESSIANS COMPLETE',flush=True)
