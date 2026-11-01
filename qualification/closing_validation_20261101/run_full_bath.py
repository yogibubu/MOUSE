"""Resume common-rule one-body confinement controls across all 23 states."""
from pathlib import Path
import socket,json,subprocess,sys,concurrent.futures,os,time
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
HERE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-closing-validation-20261101');REPAIR=HERE.parent/'MOUSE-pi-pi-transfer-repair-20261101';cases=json.loads((REPAIR/'cases.json').read_text())+[dict(cohort='closing_blind',molecule=n) for n in ['Formaldehyde','Ethylene','Butadiene']]
def worker(item):
 cohort,name=item['cohort'],item['molecule'];folder=HERE/'full_bath_screening'/cohort/name;folder.mkdir(parents=True,exist_ok=True)
 if (folder/'result.json').exists():return dict(cohort=cohort,molecule=name,status='completed_reused')
 with (folder/'driver.log').open('a') as log:code=subprocess.call([sys.executable,str(HERE/'full_bath_screening.py'),cohort,name],stdout=log,stderr=subprocess.STDOUT,env=os.environ)
 print(cohort,name,code,flush=True);return dict(cohort=cohort,molecule=name,status='completed' if code==0 else 'needs_diagnosis',returncode=code)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(worker,cases))
(HERE/'full_bath_screening/completion.json').write_text(json.dumps(results,indent=2)+'\n')
