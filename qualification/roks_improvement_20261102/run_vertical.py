"""Bounded compute-host orchestration with immutable completed case reuse."""
import socket,subprocess,sys,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
assert socket.getfqdn().lower()=='mac-2.homenet.telecomitalia.it'
r=Path(__file__).resolve().parent;p=json.loads((r/'protocol.json').read_text())
role=sys.argv[1] if len(sys.argv)>1 else 'pilot'
work=[(case['id'],f) for case in p['cases'] if case['role']==role for f in ['WB97X','CAM-B3LYP','PBE0']]
def run(pair):
 identity,functional=pair;log=r/(identity+'_'+functional+'.log')
 with log.open('a') as out:status=subprocess.run([sys.executable,str(r/'vertical.py'),identity,functional],stdout=out,stderr=subprocess.STDOUT).returncode
 row=dict(id=identity,functional=functional,returncode=status);print(json.dumps(row),flush=True);return row
with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(run,work))
(r/(role+'_completion.json')).write_text(json.dumps(rows,indent=2)+'\n')
