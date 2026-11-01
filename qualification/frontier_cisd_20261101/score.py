"""Read-only scoring after predictions; no reference enters model builders."""
from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parent
LABELS={'I001':2.403,'I002':3.715,'C001':5.489,'C002':6.027,'I004':4.606,'I005':5.341,'N001':4.304,'N002':5.644,'N003':4.144,'V001':4.468,'V002':2.877}
protocol=json.loads((ROOT/'protocol.json').read_text());rows=[]
for cohort,name in protocol['cases']+[['prospective_cisd',r['molecule']] for r in protocol['prospective_cases']]:
    folder=ROOT/cohort/name;pred=json.loads((folder/'result.json').read_text());tdareport=json.loads((folder/'tda_control.json').read_text());tda={r['id']:r for r in tdareport['predictions']}
    for p in pred['predictions']:
        reference=LABELS[p['id']];t=tda[p['id']];rows.append(dict(id=p['id'],molecule=name,cohort=cohort,reference_ev=reference,OO_ev=p['gap_ev'],OO_error_ev=p['gap_ev']-reference,TDA_ev=t['excitation_ev'],TDA_error_ev=t['excitation_ev']-reference,OO_qualified=p['qualified'],TDA_qualified=t['qualified'],native_MO_overlap=tdareport['minimum_native_MO_overlap'],auxiliary_SI_ground_lowering_ev=pred['auxiliary_si_ground_lowering_ev']))
def stats(rows,method):
    errors=[r[method+'_error_ev'] for r in rows]
    return dict(count=len(errors),MAE_ev=sum(abs(x) for x in errors)/len(errors),max_absolute_error_ev=max(abs(x) for x in errors),MSE_ev=sum(errors)/len(errors),RMSE_ev=(sum(x*x for x in errors)/len(errors))**.5,all_states_assigned=all(r[method+'_qualified'] for r in rows))
summary={group:{m:stats(rs,m) for m in ['OO','TDA']} for group,rs in [('all',rows),('exposed_development',[r for r in rows if r['cohort']!='prospective_cisd']),('prospective',[r for r in rows if r['cohort']=='prospective_cisd'])]}
gate=protocol['accuracy_gate_ev'];summary['fixed_gate_passed']=summary['all']['OO']['MAE_ev']<=gate['mae'] and summary['all']['OO']['max_absolute_error_ev']<=gate['max_absolute_error']
summary['disclosure']='Nine development and two prospective predictions on historically exposed molecules; not a truly blind validation. Primary excitation energies reference the independently stationary ground functional, not the auxiliary SI lowest eigenvalue. No fitted excitation offset.'
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (ROOT/'predictions.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
print(json.dumps(summary,indent=2))
