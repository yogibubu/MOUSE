"""Resume checkpointed chlorophyllide targets after carotene CI workers finish.

Large prepared arrays make simultaneous pigment jobs memory intensive.
This one-shot compute-host driver serializes targets and preserves every
completed certificate. It never prepares a new DFT reference or restarts
an already certified orbital target.
"""
from pathlib import Path
import socket,time,json,subprocess,os,datetime
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
HERE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-closing-validation-20261101');os.chdir(HERE)
plan=HERE/'pigment_serial_queue.json'
def status(**data):
 data.update(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=socket.getfqdn());temp=plan.with_suffix('.tmp.json');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(plan)
status(stage='Waiting for carotene convergence sequence')
while not all((HERE/f'pigment_cid_expansion/Beta_carotene/screening_8/frontier_{n}/result.json').exists() for n in [6,7,8]):time.sleep(20)
for target in [1,2,3]:
 certificate=HERE/f'pigment_roks/Chlorophyllide_a/state_{target}/certificate.json'
 if certificate.exists():continue
 status(stage='Resuming chlorophyllide orbital target',target=target)
 with open(HERE/f'chlorophyllide_serial_state_{target}.log','a') as log:
  child=subprocess.Popen(['/Users/vincenzobarone/.venvs/matrix/bin/python','-u',str(HERE/'run_pigment_roks.py'),'Chlorophyllide_a','--state',str(target)],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,VECLIB_MAXIMUM_THREADS='8'));status(stage='Running chlorophyllide orbital target',target=target,pid=child.pid);code=child.wait()
 if code:
  status(stage='Orbital target requires further solver work',target=target,exit_code=code,checkpoints_preserved=True);raise SystemExit(code)
status(stage='All checkpointed orbital targets completed; existing screening/CI workers continue')
