"""Checkpoint-preserving bounded CI expansion on the verified compute host.

At most two CI jobs coexist with the checkpointed chlorophyllide worker.
Use --existing-frontier/--existing-pid to adopt an already running worker;
only its controller is replaced, never its numerical process or checkpoints.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import subprocess,sys,socket,json,time,argparse,os
HERE=Path(__file__).resolve().parent
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
p=argparse.ArgumentParser();p.add_argument('--existing-frontier',type=int);p.add_argument('--existing-pid',type=int);a=p.parse_args()
protocol=json.loads((HERE/'beta_convergence_protocol.json').read_text())
def invoke(n):
 folder=HERE/f'pigment_cid_expansion/Beta_carotene/screening_12/frontier_{n}'
 if n==a.existing_frontier:
  assert a.existing_pid
  while not (folder/'result.json').exists():
   os.kill(a.existing_pid,0);time.sleep(10)
 else:
  with (HERE/f'beta_W12_N{n}.log').open('a') as stream:
   subprocess.run([sys.executable,'-u','run_pigment_cid.py','Beta_carotene','--frontier',str(n),'--screening-frontier','12','--roots','50','--wait-for-screening'],cwd=HERE,stdout=stream,stderr=subprocess.STDOUT,check=True)
 report=json.loads((folder/'result.json').read_text())
 return dict(frontier=n,lowest_states=report['states'][:3])
completed=[]
with ThreadPoolExecutor(max_workers=2) as pool:
 futures=[pool.submit(invoke,n) for n in (10,11,12)]
 for future in as_completed(futures):
  completed.append(future.result());(HERE/'beta_convergence_progress.json').write_text(json.dumps(dict(completed=completed,protocol=protocol),indent=2))
print('Requested windows completed; convergence must be assessed against frozen protocol.',flush=True)
