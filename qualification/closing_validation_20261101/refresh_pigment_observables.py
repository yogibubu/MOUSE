"""Correct observable phases from immutable electronic checkpoints, without SCF."""
from pathlib import Path
import json,socket,subprocess,sys
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
HERE=Path(__file__).resolve().parent
paths=list((HERE/'pigment_cid_expansion/Beta_carotene').glob('**/result.json'))+[HERE/'pigment_cid/Beta_carotene/result.json']
for path in paths:
 if not path.exists():continue
 r=json.loads(path.read_text())
 if 'observable_phase_convention' in r:continue
 n=r.get('frontier_occupied',2);w=r.get('screening_active_frontier',5)
 with (HERE/f'beta_phase_refresh_{n}_{w}.log').open('a') as stream:
  subprocess.run([sys.executable,'-u','run_pigment_cid.py','Beta_carotene','--frontier',str(n),'--screening-frontier',str(w),'--refresh-observables'],cwd=HERE,stdout=stream,stderr=subprocess.STDOUT,check=True)
 print('Refreshed observables; CI checkpoint preserved',n,w,flush=True)
