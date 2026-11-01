"""Independent fresh-process ground-engine/spin/restart checks, two remote jobs."""
from pathlib import Path
import sys,json,subprocess,concurrent.futures,socket
assert socket.getfqdn()=='oracle.sns.it'
HERE=Path('/home/enzo/MOUSE-pi-pi-transfer-repair-20261101')
def run(case):
 cohort,name=case['cohort'],case['molecule']
 if name=='Criegee_Intermediate':return
 destination=HERE/'portable_validation'/cohort/name
 report=destination/'energy.json'
 if report.exists() and json.loads(report.read_text()).get('spin'):return
 with (HERE/f'public_energy_{name}.log').open('w') as log:subprocess.run([sys.executable,str(HERE/'validate_screened.py'),cohort,name,'--spin'],stdout=log,stderr=subprocess.STDOUT,check=True)
 print(name,'energy/spin/restart passed',flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:list(executor.map(run,json.loads((HERE/'cases.json').read_text())))
