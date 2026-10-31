"""Serial Oracle controls for the separately recorded projected-Fock candidate."""
from pathlib import Path
import socket,sys,json,hashlib,subprocess,os,fcntl
assert socket.getfqdn()=='oracle.sns.it'
R=Path('/home/enzo/MOUSE-state-following-20261031');D=R/'ritz_projection_probe'
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
lock=(Path('/home/enzo/MOUSE-publication-validation-20261030/blind/references/compute.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def write(p,v):t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
plan=dict(host=socket.getfqdn(),cases=[['Bimane_Syn','B015'],['Diazirine','B001'],['Furan','B022']],weights=[.5,.5],lambda_exchange=.9,energy_offset_ev=0.,threads=2,concurrent_workers=1,reference_excitation_labels_read=False,source_sha256={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ['build_ritz_reference.py','ritz_projection_worker.py','ritz_projection_audit.py','validate_ritz_controls.py']},scope='Bimane method repair plus two previously completed positive controls; exploratory numerical validation, not blind accuracy validation')
p=D/'controls_protocol.json'
if p.exists():assert json.loads(p.read_text())==plan
else:write(p,plan)
def launch(source,args):
 with (D/'controls.log').open('a') as f:subprocess.run([sys.executable,str(R/source)]+args,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=3600)
for name,sid in plan['cases']:
 out=D/'results'/name/sid/'result.json'
 if not out.exists():
  write(D/'progress.json',dict(status='running',molecule=name,state=sid))
  launch('build_ritz_reference.py',[name]);launch('ritz_projection_worker.py',[name,sid])
 launch('ritz_projection_audit.py',[])
 result=json.loads(out.read_text());assert result['status']=='matched',result
rows=[json.loads((D/'results'/n/s/'result.json').read_text()) for n,s in plan['cases']]
write(D/'validation_report.json',dict(host=socket.getfqdn(),status='passed',cases=rows,accuracy_scoring_performed=False,limitation='Bimane reference subset contains only the assigned root; competitor exclusivity is not tested there',reference_excitation_labels_read=False))
write(D/'progress.json',dict(status='passed',cases=len(rows)))
