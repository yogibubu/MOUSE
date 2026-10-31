"""Unseal labels only after the complete six-state prediction ledger closes."""
from pathlib import Path
import json,hashlib,socket,datetime,math
assert socket.getfqdn()=='oracle.sns.it'
R=Path(__file__).resolve().parent;root=R/'independent';manifest=json.loads((root/'manifest.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[]
for row in manifest['states']:
 file=root/'predictions'/row['molecule']/row['id']/'result.json'
 if not file.exists():raise SystemExit('Prediction ledger remains open; sealed labels were NOT read: '+row['id'])
 predicted=json.loads(file.read_text());assert predicted['status'] in ['matched','not_representable','nonconverged','completed_unassigned'];rows.append((row,predicted,file))
lock=dict(closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest_sha256=sha(root/'manifest.json'),predictions_sha256={str(p.relative_to(root)):sha(p) for _,_,p in rows},candidate_frozen=True)
f=root/'prediction_lock.json'
if f.exists():assert json.loads(f.read_text())['predictions_sha256']==lock['predictions_sha256'],'Prediction ledger changed after scoring lock'
else:f.write_text(json.dumps(lock,indent=2)+'\n')
labels_file=root/'sealed_labels.json';assert sha(labels_file)==manifest['sealed_labels_sha256'];labels=json.loads(labels_file.read_text());scored=[]
for row,predicted,path in rows:
 result=dict(predicted,reference_ev=labels[row['id']]['tbe_ev'],reference_safe=labels[row['id']]['safe'],partial_double_character=row['partial_double_character'])
 if predicted['status']=='matched':result['error_ev']=predicted['excitation_ev']-float(result['reference_ev'])
 tdafile=root/'tda'/row['molecule']/'result.json'
 if tdafile.exists():
  controls=json.loads(tdafile.read_text());td=next((r for r in controls['rows'] if r['id']==row['id']),None);result['tda_gfn2']=td
  if td is not None:
   for parent in td.get('parents',[]):
    if parent['status']=='matched':parent['error_ev']=parent['excitation_ev']-float(result['reference_ev'])
 scored.append(result)
def metrics(values):
 if not values:return dict(n=0)
 return dict(n=len(values),mae=sum(abs(v) for v in values)/len(values),mse=sum(values)/len(values),rmse=math.sqrt(sum(v*v for v in values)/len(values)),maxabs=max(abs(v) for v in values))
report=dict(schema='mouse.independent.six.score.v1',n_attempted=len(scored),n_matched=sum(r['status']=='matched' for r in scored),overall=metrics([r['error_ev'] for r in scored if 'error_ev' in r]),families={family:metrics([r['error_ev'] for r in scored if r['family']==family and 'error_ev' in r]) for family in ['n_pi','pi_pi','Rydberg','CT']},rows=scored,limitation='six predeclared states, five molecules; CT shortfall; failed representation remains in denominator; no parameter refitting')
(root/'score.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(attempted=report['n_attempted'],matched=report['n_matched'],metrics=report['overall'])))
