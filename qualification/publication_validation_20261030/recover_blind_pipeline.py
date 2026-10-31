"""Fail-fast setup recovery plus serial targeted-reference/MOUSE continuation."""
from pathlib import Path
import socket,os,sys,json,hashlib,fcntl,subprocess,signal,datetime
assert socket.getfqdn()=='oracle.sns.it'
P=Path(__file__).resolve().parent
import blind_mouse as mouse
from serial_blind_pipeline import write,progress

def run(kind,row):
 name=row['molecule'];d=P/'blind'/('mouse' if kind=='mouse' else 'references')/name;d.mkdir(exist_ok=True)
 file='blind_mouse.py' if kind=='mouse' else 'targeted_reference.py';args=['--worker',name] if kind=='mouse' else [name]
 with (d/'recovery_20261031.log').open('a') as log:
  proc=subprocess.Popen(['/usr/bin/time','-v','-o',str(d/'recovery_20261031.resources'),sys.executable,str(P/file)]+args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
  progress('mouse' if kind=='mouse' else 'targeted_reference',name,pid=proc.pid)
  if kind!='mouse':write(d.parent/'progress.json',dict(status='running',molecule=name,pid=proc.pid,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reference_mode='targeted symmetries',numerical_labels_read=False))
  try:rc=proc.wait(timeout=3600)
  except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();rc=-9
 if rc:
  write(d/'recovery_20261031.failure.json',dict(status='timeout' if rc==-9 else 'execution_error',returncode=rc,checkpoint_preserved=True))
  if kind=='mouse':raise RuntimeError('MOUSE worker failed; stop orchestration for diagnosis: '+name)
 return rc
lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert json.loads((P/'blind/mouse/protocol.json').read_text())==mouse.get_plan()
plan=dict(host=socket.getfqdn(),date='2026-10-31',reason='40-root reference responses exceeded time budget; use fixed public target symmetry and ordinal only',unchanged=['PBE0/aug-cc-pVDZ','SCF and TDA convergence thresholds','two threads, one electronic worker','MOUSE source and frozen energy model','public assignment catalog, state manifest and scoring labels'],ground_checkpoints='reuse all completed ground calculations',previous_spectra='preserve all completed and partial spectra',new_reference_response='target wfnsym; number of roots is largest frozen ordinal in that symmetry; record unsupported group correlations',post_score_fitting=False,source_sha256={n:mouse.digest(P/n) for n in ['targeted_reference.py','recover_blind_pipeline.py','blind_mouse.py']},assignment_catalog_sha256=mouse.digest(P/'blind/assignment_catalog.json'))
f=P/'blind/targeted_protocol.json'
if f.exists():assert json.loads(f.read_text())==plan
else:write(f,plan)
rows={}
for r in json.loads((P/'blind/manifest.json').read_text())['states']:rows.setdefault(r['molecule'],r)
ordered=sorted(rows.values(),key=lambda r:(r['n_atoms'],r['molecule']))
for r in ordered:
 if (P/'blind/references'/r['molecule']/'result.json').exists() and not (P/'blind/mouse'/r['molecule']/'result.json').exists():run('mouse',r)
for r in ordered:
 d=P/'blind/references'/r['molecule']
 if (d/'result.json').exists():continue
 rc=run('reference',r)
 if rc==0 and (d/'result.json').exists():run('mouse',r)
progress('recovery_pass_complete',None,labels_unsealed=False)
