"""Common spectral-block amendment on all 26 states; no parameter refit."""
from pathlib import Path
import socket,sys,subprocess,json,os,concurrent.futures
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
HERE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-closing-validation-20261101');paths=sorted((HERE/'disentangled_bath_eta_0.600000').glob('*/*/result.json'))
def worker(path):
 cohort,name=path.parent.parent.name,path.parent.name;out=HERE/'spectral_block_reference'/cohort/name;out.mkdir(parents=True,exist_ok=True)
 with (out/'driver.log').open('a') as log:
  code=subprocess.call([sys.executable,str(HERE/'spectral_block_reference.py'),cohort,name],stdout=log,stderr=subprocess.STDOUT,env=os.environ)
  if code==0:code=subprocess.call([sys.executable,str(HERE/'disentangled_bath_screening.py'),cohort,name,'--bath-strength','.60','--reference-parent','spectral_block_reference'],stdout=log,stderr=subprocess.STDOUT,env=os.environ)
 row=dict(cohort=cohort,molecule=name,returncode=code);(out/'status.json').write_text(json.dumps(row,indent=2)+'\n');print(row,flush=True);return row
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(worker,paths))
(HERE/'spectral_block_reference/completion.json').write_text(json.dumps(rows,indent=2)+'\n')
