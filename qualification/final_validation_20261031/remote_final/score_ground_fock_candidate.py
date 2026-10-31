"""Lock every frozen ground-Fock prediction before scoring; no parameter selection."""
from pathlib import Path
import json,hashlib,socket,math,datetime
assert socket.getfqdn()=='oracle.sns.it'
R=Path(__file__).resolve().parent;root=R/'ground_fock_candidate';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();ledgers={}
for cohort in ['independent','independent_challenge']:
 manifest=json.loads((R/cohort/'manifest.json').read_text());rows=[]
 for row in manifest['states']:
  p=root/cohort/row['molecule']/row['id']/'result.json'
  if not p.exists():raise SystemExit('Prediction remains open; labels not read: '+cohort+'/'+row['id'])
  d=json.loads(p.read_text());assert d['status'] in ['matched','not_representable'];rows.append((row,d,p))
 ledgers[cohort]=(manifest,rows)
lock=dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidate_protocol_sha256=sha(R/'ground_fock_candidate_protocol.json'),predictions_sha256={str(p.relative_to(root)):sha(p) for manifest,rows in ledgers.values() for row,d,p in rows},scope='Candidate specification frozen before primary score existed; predictions locked before this scorer opens numerical labels. Furofuran is exposed development; higher benzoxadiazole labels had already been exposed.')
p=root/'prediction_lock.json'
if p.exists():assert json.loads(p.read_text())['predictions_sha256']==lock['predictions_sha256']
else:p.write_text(json.dumps(lock,indent=2)+'\n')
def metrics(e):return dict(n=len(e),mae=sum(abs(v) for v in e)/len(e),mse=sum(e)/len(e),rmse=math.sqrt(sum(v*v for v in e)/len(e)),maxabs=max(abs(v) for v in e)) if e else dict(n=0)
for cohort,(manifest,rows) in ledgers.items():
 labels_file=R/cohort/'sealed_labels.json';assert sha(labels_file)==manifest['sealed_labels_sha256'];labels=json.loads(labels_file.read_text());baseline=json.loads((R/cohort/'score.json').read_text());scored=[]
 for row,d,p in rows:
  value=dict(d,family=row['family'],reference_ev=labels[row['id']]['tbe_ev'],partial_double_character=row['partial_double_character'])
  if d['status']=='matched':value['error_ev']=d['excitation_ev']-value['reference_ev']
  value['baseline']=next(r for r in baseline['rows'] if r['id']==row['id']);scored.append(value)
 report=dict(cohort=cohort,n_attempted=len(scored),n_matched=sum(r['status']=='matched' for r in scored),overall=metrics([r['error_ev'] for r in scored if 'error_ev'in r]),rows=scored,candidate_protocol_sha256=lock['candidate_protocol_sha256'],parameters_selected_from_scores=False)
 (root/cohort/'score.json').write_text(json.dumps(report,indent=2)+'\n');print(cohort,report['overall'])
