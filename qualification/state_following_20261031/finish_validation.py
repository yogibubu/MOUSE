"""Serial Oracle validation after the currently running larger recovery finishes."""
from pathlib import Path
import socket,os,sys,json,time,subprocess,datetime
assert socket.getfqdn()=='oracle.sns.it'
R=Path('/home/enzo/MOUSE-state-following-20261031');os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MOUSE_VALIDATION_ROOT=str(R))
pid=int(sys.argv[1]);deadline=time.monotonic()+3600
while Path(f'/proc/{pid}/cmdline').exists():
 cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
 if b'continuation_rank_one.py' not in cmd:break
 if time.monotonic()>deadline:raise RuntimeError('Previous recovery still active; refuse concurrent electronic work')
 time.sleep(5)
steps=[('library_diazirine',[sys.executable,str(R/'check_library_continuation.py'),'--molecule','Diazirine','--state','B001'],R),('library_cyclopentadiene',[sys.executable,str(R/'check_library_continuation.py'),'--molecule','Cyclopentadiene','--state','B016'],R),('regression_tests',[sys.executable,'-c','import remote_bootstrap,pytest; raise SystemExit(pytest.main(["test_roks.py","test_roks_tracking.py","test_roks_trial_guard.py","test_orbital_saddle.py","test_relative_pair_roks.py","-q"]))'],R/'source/qualification'),('assignment_audit',[sys.executable,str(R/'audit_recovered_assignment.py')],R)]
rows=[]
for name,args,cwd in steps:
 status=dict(stage=name,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=socket.getfqdn())
 (R/'validation_progress.json').write_text(json.dumps(status,indent=2)+'\n');print(status,flush=True)
 with (R/(name+'.log')).open('w') as log:
  run=subprocess.run(['/usr/bin/time','-v','-o',str(R/(name+'.resources'))]+args,cwd=cwd,env=os.environ,stdout=log,stderr=subprocess.STDOUT,timeout=900)
 rows.append(dict(stage=name,returncode=run.returncode))
 if run.returncode:
  (R/'validation_progress.json').write_text(json.dumps(dict(status='failed',rows=rows),indent=2)+'\n');raise SystemExit(run.returncode)
(R/'validation_progress.json').write_text(json.dumps(dict(status='passed',rows=rows,host=socket.getfqdn(),labels_read=False),indent=2)+'\n')
print('VALIDATION PASSED',rows,flush=True)
