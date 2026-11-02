"""Fresh public-API checks for all existing molecular models, no refitting."""
from pathlib import Path
import socket,sys,subprocess,json,concurrent.futures
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
HERE=Path(__file__).resolve().parent
paths=sorted((HERE/'disentangled_bath_spectral_block_reference_eta_0.600000').glob('*/*/model.json'))
def worker(path):
 cohort,name=path.parent.parent.name,path.parent.name;out=HERE/'static_subtraction_models'/cohort/name;out.mkdir(parents=True,exist_ok=True)
 with (out/'qualification.log').open('a') as log:
  result=subprocess.run([sys.executable,'-u',str(HERE/'qualify_static_model.py'),cohort,name],cwd=HERE,stdout=log,stderr=subprocess.STDOUT)
 row=dict(cohort=cohort,molecule=name,returncode=result.returncode);print(row,flush=True);return row
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(worker,paths))
(HERE/'static_subtraction_models/completion.json').write_text(json.dumps(rows,indent=2)+'\n')
if any(r['returncode'] for r in rows):raise RuntimeError('Public API candidate validation incomplete')
