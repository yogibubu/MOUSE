"""Checkpoint-preserving common-rule complete-ledger candidate evaluation."""
from pathlib import Path
import socket,json,subprocess,sys,concurrent.futures,os
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
assert subprocess.check_output(['scutil','--get','ComputerName'],text=True).strip()=='Mac Studio di Vincenzo'
HERE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-closing-validation-20261101');REPAIR=HERE.parent/'MOUSE-pi-pi-transfer-repair-20261101'
cases=json.loads((REPAIR/'cases.json').read_text())+[dict(cohort='closing_blind',molecule=n) for n in ['Formaldehyde','Ethylene','Butadiene']]
groups={};seen=set()
for case in cases:
 key=case['cohort'],case['molecule']
 if key not in seen:groups.setdefault(key[0],[]).append(key);seen.add(key)
def group(items):
 rows=[]
 for cohort,name in items:
  out=HERE/'subspace_cisd_driver'/cohort/name;out.mkdir(parents=True,exist_ok=True)
  with (out/'driver.log').open('a') as log:
   code=subprocess.call([sys.executable,str(HERE/'radial_embedding.py'),cohort,name,'--mode','subspace'],stdout=log,stderr=subprocess.STDOUT,env=os.environ)
   if code==0:code=subprocess.call([sys.executable,str(HERE/'complete_singles.py'),'subspace_'+cohort,name,'--parent','original_prepared'],stdout=log,stderr=subprocess.STDOUT,env=os.environ)
  row=dict(cohort=cohort,molecule=name,returncode=code);rows.append(row);(out/'status.json').write_text(json.dumps(row,indent=2)+'\n');print(row,flush=True)
 return rows
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=[row for result in pool.map(group,groups.values()) for row in result]
(HERE/'subspace_cisd_driver/completion.json').write_text(json.dumps(rows,indent=2)+'\n')
