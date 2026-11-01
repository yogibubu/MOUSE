"""Score already completed immutable predictions; no electronic calculations."""
from pathlib import Path
import json,csv,statistics
ROOT=Path(__file__).resolve().parent
refs={'I001':2.403,'I002':3.715,'C001':5.489,'C002':6.027,'N001':4.304,'N002':5.644,'N003':4.144,'I004':4.606,'I005':5.341,'V001':4.468,'V002':2.877,'V003':2.462,'V004':3.723,'V005':1.952,'V006':3.810}
rows=[]
for path in sorted((ROOT/'results/selected_cid').glob('*/*/result.json')):
 cohort,name=path.parts[-3:-1];tda=json.loads((ROOT/'results/tda'/cohort/name/'tda_control.json').read_text());controls={r['id']:r for r in tda['predictions']}
 for p in json.loads(path.read_text())['primary']['predictions']:
  ref=refs[p['id']];t=controls[p['id']]
  assert p['qualified'] and t['qualified']
  rows.append(dict(id=p['id'],cohort=cohort,molecule=name,reference_ev=ref,OO_ev=p['gap_ev'],TDA_ev=t['excitation_ev'],OO_error_ev=p['gap_ev']-ref,TDA_error_ev=t['excitation_ev']-ref,assignment_overlap=p['overlap'],assignment_margin=p['margin'],TDA_wall_seconds=t['wall_seconds'],native_MO_overlap=tda['minimum_native_MO_overlap']))
assert len(rows)==15
rows.sort(key=lambda r:(r['cohort']=='cid_prospective',r['id']))
summary={}
for label,subset in [('all',rows),('development',[r for r in rows if r['cohort']!='cid_prospective']),('prospective',[r for r in rows if r['cohort']=='cid_prospective'])]:
 summary[label]=dict(states=len(subset),OO_MAE_ev=statistics.mean(abs(r['OO_error_ev']) for r in subset),OO_max_ev=max(abs(r['OO_error_ev']) for r in subset),TDA_MAE_ev=statistics.mean(abs(r['TDA_error_ev']) for r in subset),TDA_max_ev=max(abs(r['TDA_error_ev']) for r in subset))
summary.update(candidate_freeze_commit='467a9e5',weights=[.5,.5],lambda_exchange=.9,energy_offset_ev=0.,blind=False,exposure='13 development states; 2 predictions after candidate freeze on molecules historically exposed to older models',all_fixed_accuracy_gates_pass=summary['all']['OO_MAE_ev']<=.3 and summary['all']['OO_max_ev']<=.5)
(ROOT/'results/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (ROOT/'results/predictions.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
print(json.dumps(summary,indent=2))
