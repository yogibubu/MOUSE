"""Two molecules concurrently, sequential qualified functional continuation."""
import socket,subprocess,sys,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
assert socket.getfqdn().lower()=='mac-2.homenet.telecomitalia.it'
r=Path(__file__).resolve().parent;p=json.loads((r/'protocol.json').read_text())
role=sys.argv[1] if len(sys.argv)>1 else 'pilot'
work=[case['id'] for case in p['cases'] if case['role']==role]
def run(identity):
 rows=[]
 # Execution order changes initial orbitals only, never model ranking rules.
 for functional in ['CAM-B3LYP','WB97X','PBE0']:
  log=r/(identity+'_'+functional+'.log')
  with log.open('a') as out:status=subprocess.run([sys.executable,str(r/'vertical.py'),identity,functional],stdout=out,stderr=subprocess.STDOUT).returncode
  row=dict(id=identity,functional=functional,returncode=status);print(json.dumps(row),flush=True);rows.append(row)
 return rows
with ThreadPoolExecutor(max_workers=2) as pool:rows=[row for group in pool.map(run,work) for row in group]
(r/(role+'_completion.json')).write_text(json.dumps(rows,indent=2)+'\n')
subprocess.run([sys.executable,str(r/'score_vertical.py')],check=True)
