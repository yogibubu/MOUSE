"""Audit every completed common-rule model in a fresh DFT-free process."""
from pathlib import Path
import socket,subprocess,sys,concurrent.futures,os,json
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
HERE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-closing-validation-20261101');variant='disentangled_bath_spectral_block_reference_eta_0.600000'
paths=sorted((HERE/variant).glob('*/*/result.json'))
def worker(p):
 cohort,name=p.parent.parent.name,p.parent.name
 with (p.parent/'runtime_driver.log').open('a') as log:code=subprocess.call([sys.executable,str(HERE/'audit_prepared_energy.py'),cohort,name,'--variant',variant],stdout=log,stderr=subprocess.STDOUT,env=os.environ)
 print(cohort,name,code,flush=True);return dict(cohort=cohort,molecule=name,returncode=code)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(worker,paths))
(HERE/variant/'runtime_completion.json').write_text(json.dumps(rows,indent=2)+'\n')
