"""Fixed-model prospective extension; numerical reference labels never loaded."""
from pathlib import Path
import socket,subprocess,sys,os,json,concurrent.futures,datetime,hashlib
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
assert subprocess.check_output(['scutil','--get','ComputerName'],text=True).strip()=='Mac Studio di Vincenzo'
BASE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101');HERE=BASE/'oracle/MOUSE-closing-validation-20261101';protocol=json.loads((HERE/'additional_protocol.json').read_text());cohort=protocol['cohort'];manifest=BASE/'oracle/MOUSE-cam-response-20261101/data'/cohort/'manifest.json';manifest.parent.mkdir(parents=True,exist_ok=True)
if not manifest.exists():manifest.write_text(json.dumps(dict(states=[dict(id=c['id'],molecule=c['molecule'],ordinal_same_symmetry=1,symmetry='HOMO-LUMO assigned',family='unscored') for c in protocol['cases']],target_energy_labels_included=False),indent=2)+'\n')
def worker(case):
 name,identity=case['molecule'],case['id'];dest=HERE/'additional_frozen'/name;dest.mkdir(parents=True,exist_ok=True)
 stages=[['prepare_fresh.py',name,identity,str(HERE/'additional_inputs'/(name+'.xyz')),'--cohort',cohort],['downfold_reference.py',cohort,name],['disentangled_bath_screening.py',cohort,name,'--bath-strength','0.60'],['prepared_matched_tda.py',cohort,name],['audit_prepared_energy.py',cohort,name]]
 for stage in stages:
  with (dest/'driver.log').open('a') as log:code=subprocess.call([sys.executable,str(HERE/stage[0]),*stage[1:]],stdout=log,stderr=subprocess.STDOUT,env=os.environ)
  if code:break
 result=dict(molecule=name,id=identity,returncode=code,stage=stage[0],completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_sha256=hashlib.sha256((HERE/'additional_protocol.json').read_bytes()).hexdigest(),reference_labels_read=False)
 predicted=HERE/'disentangled_bath_eta_0.600000'/cohort/name/'result.json'
 if predicted.exists():result.update(prediction_sha256=hashlib.sha256(predicted.read_bytes()).hexdigest(),prediction=json.loads(predicted.read_text()))
 (dest/'status.json').write_text(json.dumps(result,indent=2)+'\n');print(name,result['returncode'],result['stage'],flush=True);return result
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(worker,protocol['cases']))
(HERE/'additional_frozen/completion.json').write_text(json.dumps(results,indent=2)+'\n')
