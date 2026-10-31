"""Finite, serial follow-up to the six-state frozen prediction pass."""
from pathlib import Path
import socket,json,time,subprocess,sys,hashlib,datetime
assert socket.getfqdn()=='oracle.sns.it'
R=Path(__file__).resolve().parent;root=R/'independent';manifest=json.loads((root/'manifest.json').read_text());start=time.monotonic()
def write(p,d):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
write(R/'postprocess_execution.json',dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in R.glob('*.py')},sealed_labels_read=False))
while time.monotonic()-start<14400:
 missing=[r['id'] for r in manifest['states'] if not (root/'predictions'/r['molecule']/r['id']/'result.json').exists()]
 if not missing:break
 write(R/'postprocess_progress.json',dict(status='waiting_for_closed_prediction_ledger',missing=missing,labels_read=False));time.sleep(10)
else:raise SystemExit('Prediction pass incomplete; sealed labels remain closed')
for name in dict.fromkeys(r['molecule'] for r in manifest['states']):
 write(R/'postprocess_progress.json',dict(status='computing_matched_gfn2_tda_controls',molecule=name,labels_read=False))
 subprocess.run([sys.executable,str(R/'independent_controls.py'),name],cwd=R,check=True,timeout=7200)
subprocess.run([sys.executable,str(R/'score_independent.py')],cwd=R,check=True)
if (R/'geometry/Furan/result.json').exists():subprocess.run([sys.executable,str(R/'compare_furan.py'),str(R)],cwd=R,check=True)
subprocess.run([sys.executable,str(R/'collect.py')],cwd=R,check=True)
write(R/'postprocess_progress.json',dict(status='independent_scored',all_six_attempts_closed=True,labels_read_only_after_prediction_lock=True,cost_and_other_geometry_status='see separate controller ledger'))
