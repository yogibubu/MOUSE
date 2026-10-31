from pathlib import Path
import subprocess,sys,time,socket
assert socket.getfqdn()=='oracle.sns.it'
R=Path(__file__).resolve().parent
for attempt in range(120):
 result=subprocess.run([sys.executable,str(R/'score_ground_fock_candidate.py')],cwd=R,capture_output=True,text=True)
 if result.returncode==0:print(result.stdout,flush=True);break
 time.sleep(20)
else:raise SystemExit('Frozen prediction ledger not closed; no score produced by this runner')
