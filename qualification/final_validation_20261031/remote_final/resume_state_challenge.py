"""Reuse completed higher states and recover the blocked rank-two Q parent."""
from pathlib import Path
import socket,os,sys,time,subprocess,json,signal
assert socket.getfqdn()=='oracle.sns.it'
R=Path(__file__).resolve().parent;pid=int(sys.argv[1]);start=time.monotonic()
while Path(f'/proc/{pid}').exists() and time.monotonic()-start<3600:time.sleep(2)
assert not Path(f'/proc/{pid}').exists(),'Prior worker remains active; do not duplicate it'
p=R/'independent_state_challenge/predictions/Benzoxadiazole/S001/result.json'
if p.exists() and json.loads(p.read_text())['status']!='matched':p.rename(p.with_name('fixed_initial_index_result.json'))
env=dict(os.environ,MOUSE_HOLDOUT_DIRECTORY='independent_state_challenge')
result=subprocess.run([sys.executable,str(R/'independent.py'),'Benzoxadiazole'],cwd=R,env=env)
if p.exists() and json.loads(p.read_text())['status']=='matched':
 try:os.kill(1028390,signal.SIGCONT)
 except ProcessLookupError:subprocess.Popen([sys.executable,str(R/'postprocess.py')],cwd=R,env=env)
