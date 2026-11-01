"""Adopt a costly in-flight trial, then resume its accepted checkpoint with the faster proposal."""
from pathlib import Path
import socket,json,time,os,signal,subprocess,sys,argparse
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('state',type=int);p.add_argument('pid',type=int);p.add_argument('--retract-constraint',action='store_true');p.add_argument('--moving-frame-jacobian',action='store_true');p.add_argument('--minimum-iteration',type=int,default=1);a=p.parse_args();folder=HERE/f'pigment_roks/Chlorophyllide_a/state_{a.state}'
while True:
 if (folder/'certificate.json').exists():raise SystemExit(0)
 row=json.loads((folder/'progress.json').read_text())['row']
 if row['iteration']>=a.minimum_iteration:break
 try:os.kill(a.pid,0)
 except ProcessLookupError:raise RuntimeError('In-flight worker exited before accepting a step')
 time.sleep(1)
os.kill(a.pid,signal.SIGTERM)
print('Accepted full-response trust step preserved',a.state,row,flush=True)
with (HERE/f'chlorophyllide_repair_{"moving" if a.moving_frame_jacobian else "retracted" if a.retract_constraint else "aux"}_state_{a.state}.log').open('a') as log:
 subprocess.run([sys.executable,'-u','run_pigment_roks.py','Chlorophyllide_a','--state',str(a.state),'--trust-krylov-dimension','16','--maximum-step','0.5']+(['--retract-constraint'] if a.retract_constraint else [])+(['--moving-frame-jacobian'] if a.moving_frame_jacobian else []),cwd=HERE,stdout=log,stderr=subprocess.STDOUT,check=True)
