"""Resume all exposed states with one explicitly empirical bath efficiency."""
from pathlib import Path
import socket,json,subprocess,sys,concurrent.futures,os,argparse
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
assert subprocess.check_output(['scutil','--get','ComputerName'],text=True).strip()=='Mac Studio di Vincenzo'
HERE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-closing-validation-20261101');REPAIR=HERE.parent/'MOUSE-pi-pi-transfer-repair-20261101'
p=argparse.ArgumentParser();p.add_argument('--bath-strength',type=float,default=.65);a=p.parse_args();eta=a.bath_strength
cases=json.loads((REPAIR/'cases.json').read_text())+[dict(cohort='closing_blind',molecule=n) for n in ['Formaldehyde','Ethylene','Butadiene']];unique={(r['cohort'],r['molecule']):r for r in cases}
def worker(key):
 cohort,name=key;out=HERE/f'disentangled_bath_eta_{eta:.6f}'/cohort/name;out.mkdir(parents=True,exist_ok=True)
 with (out/'driver.log').open('a') as log:code=subprocess.call([sys.executable,str(HERE/'disentangled_bath_screening.py'),cohort,name,'--bath-strength',str(eta)],stdout=log,stderr=subprocess.STDOUT,env=os.environ)
 row=dict(cohort=cohort,molecule=name,returncode=code);print(row,flush=True);return row
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(worker,unique))
(HERE/f'disentangled_bath_eta_{eta:.6f}'/'completion.json').write_text(json.dumps(rows,indent=2)+'\n')
