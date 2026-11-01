"""Score the frozen final two-case extension and aggregate without refitting."""
from pathlib import Path
import json,csv,statistics,hashlib
Q=Path(__file__).resolve().parent
ROOT=Q/'last_two_results'
protocol=json.loads((Q/'last_two_protocol.json').read_text())
refs={'V010':6.530,'V011':5.381}
with (Q/'additional_three_results/combined_predictions.csv').open() as f:baseline=list(csv.DictReader(f))
rows=[]
for case in protocol['cases']:
 name=case['molecule'];cohort=protocol['cohort'];folder=ROOT/'selected_cid'/cohort/name
 p=next(r for r in json.loads((folder/'result.json').read_text())['primary']['predictions'] if r['id']==case['id'])
 controls=json.loads((ROOT/'tda'/cohort/name/'tda_control.json').read_text());t=next(r for r in controls['predictions'] if r['id']==case['id']);ref=refs[case['id']]
 rows.append(dict(id=case['id'],cohort=cohort,molecule=name,reference_ev=ref,OO_ev=p['gap_ev'],TDA_ev=t['excitation_ev'],OO_error_ev=p['gap_ev']-ref,TDA_error_ev=t['excitation_ev']-ref,assignment_overlap=p['overlap'],assignment_margin=p['margin'],TDA_wall_seconds=t['wall_seconds'],native_MO_overlap=controls['minimum_native_MO_overlap'],OO_assigned=bool(p['qualified']),TDA_assigned=bool(t['qualified']),individual_accuracy_pass=abs(p['gap_ev']-ref)<=.5))
assert len(baseline)==18 and len(rows)==2
combined=baseline+rows;summary={}
def statistics_for(subset):
 return dict(states=len(subset),OO_MAE_ev=statistics.mean(abs(float(r['OO_error_ev'])) for r in subset),OO_max_ev=max(abs(float(r['OO_error_ev'])) for r in subset),TDA_MAE_ev=statistics.mean(abs(float(r['TDA_error_ev'])) for r in subset),TDA_max_ev=max(abs(float(r['TDA_error_ev'])) for r in subset))
summary['last_two']=statistics_for(rows);summary['all']=statistics_for(combined);summary['development']=statistics_for([r for r in combined if r['cohort'] not in {'cid_prospective','additional_three','last_two'}]);summary['prospective']=statistics_for([r for r in combined if r['cohort'] in {'cid_prospective','additional_three','last_two'}]);summary['additional_three']=statistics_for([r for r in baseline if r['cohort']=='additional_three']);summary['all_new_assignments_pass']=all(r['OO_assigned'] and r['TDA_assigned'] for r in rows)
pi_ids={'C001','C002','I002','I004','I005','V010','V011'}
summary['pi_pi']=statistics_for([r for r in combined if r['id'] in pi_ids]);summary['n_pi']=statistics_for([r for r in combined if r['id'] not in pi_ids])
for stats in summary.values():
 if isinstance(stats,dict) and 'OO_MAE_ev' in stats:stats['accuracy_gate_pass']=stats['OO_MAE_ev']<=.3 and stats['OO_max_ev']<=.5
summary.update(candidate_freeze_commit='467a9e5',cohort_freeze_commit='a474e68',weights=[.5,.5],lambda_exchange=.9,energy_offset_ev=0,model_changed=False,blind=False,exposure='13 development states plus 7 predictions after candidate freeze on historically exposed molecules',pooled_accuracy_gates_pass=summary['all']['accuracy_gate_pass'],additional_cohort_accuracy_gates_pass=summary['last_two']['accuracy_gate_pass'],all_fixed_accuracy_gates_pass=summary['all']['accuracy_gate_pass'] and summary['last_two']['accuracy_gate_pass'] and summary['additional_three']['accuracy_gate_pass'] and summary['prospective']['accuracy_gate_pass'])
for name,items in [('predictions.csv',rows),('combined_predictions.csv',combined)]:
 with (ROOT/name).open('w') as f:
  fields=list(rows[0]) if name=='predictions.csv' else list(baseline[0]);w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');w.writeheader();w.writerows(items)
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
