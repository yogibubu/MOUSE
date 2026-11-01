"""Independent fresh-process ground-engine/spin/restart checks, two remote jobs."""
from pathlib import Path
import sys,json,subprocess,concurrent.futures,socket
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
HERE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-pi-pi-transfer-repair-20261101')
def run(case):
 cohort,name=case['cohort'],case['molecule']
 destination=HERE/'portable_validation'/cohort/name
 report=destination/'energy.json'
 if report.exists() and json.loads(report.read_text()).get('runtime_sources'):return
 with (HERE/f'public_energy_{name}.log').open('w') as log:subprocess.run([sys.executable,str(HERE/'validate_screened.py'),cohort,name,'--spin'],stdout=log,stderr=subprocess.STDOUT,check=True)
 print(name,'energy/spin/restart passed',flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:list(executor.map(run,json.loads((HERE/'cases.json').read_text())))
