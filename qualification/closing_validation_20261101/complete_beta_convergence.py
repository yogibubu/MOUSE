"""Checkpoint-preserving serial CI expansion on the verified compute host."""
from pathlib import Path
import subprocess,sys,socket,json,time
HERE=Path(__file__).resolve().parent
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
protocol=json.loads((HERE/'beta_convergence_protocol.json').read_text())
def invoke(args,log):
 with (HERE/log).open('a') as stream:
  subprocess.run([sys.executable,'-u',*args],cwd=HERE,stdout=stream,stderr=subprocess.STDOUT,check=True)
for n in (10,11,12):
 invoke(['run_pigment_cid.py','Beta_carotene','--frontier',str(n),'--screening-frontier','12','--roots','50','--wait-for-screening'],f'beta_W12_N{n}.log')
 folder=HERE/f'pigment_cid_expansion/Beta_carotene/screening_12/frontier_{n}'
 report=json.loads((folder/'result.json').read_text())
 (HERE/'beta_convergence_progress.json').write_text(json.dumps(dict(completed_frontier=n,lowest_states=report['states'][:3],protocol=protocol),indent=2))
print('Requested windows completed; convergence must be assessed against frozen protocol.',flush=True)
