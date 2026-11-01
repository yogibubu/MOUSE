"""Scoring module; never imported by preparation or prediction drivers."""
from pathlib import Path
import json,hashlib
ROOT=Path('/home/enzo/MOUSE-m06-response-20261101')
REF=Path('/home/enzo/MOUSE-orbital-performance-20261031/multicsf_integration/final_validation')
cases=[('development','independent','Criegee_Intermediate'),('development','independent_challenge','Furofuran'),
       ('model_heldout','independent','Benzoxadiazole'),('model_heldout','independent','Hexatriene')]
rows=[]
for split,cohort,name in cases:
    references={r['id']:r for r in json.loads((REF/cohort/'score.json').read_text())['rows']}
    folder=ROOT/'nosi_coherent'/cohort/name
    d=json.loads((folder/'projected_result.json').read_text())
    for prediction in d['predictions']:
        identity=prediction['id'];ref=references[identity];r=json.loads((ROOT/cohort/name/'coherent_states'/identity/'result.json').read_text())
        rows.append(dict(split=split,id=identity,molecule=name,reference_ev=ref['reference_ev'],original_ev=ref['excitation_ev'],
            coherent_roks_ev=r['excitation_ev'],candidate_ev=prediction['gap_ev'],candidate_error_ev=prediction['gap_ev']-ref['reference_ev'],
            qualified=bool(prediction['qualified'] and r['qualified']),character=prediction['overlap'],kkt_max=r['kkt_max'],
            result_sha256=hashlib.sha256((folder/'projected_result.json').read_bytes()).hexdigest()))
def metrics(selected):
    result={}
    for method in ['original_ev','coherent_roks_ev','candidate_ev']:
        errors=[r[method]-r['reference_ev'] for r in selected]
        result[method]=dict(n=len(errors),mae_ev=sum(abs(x) for x in errors)/len(errors),maxabs_ev=max(abs(x) for x in errors),rmse_ev=(sum(x*x for x in errors)/len(errors))**.5)
    return result
groups={split:metrics([r for r in rows if r['split']==split]) for split in ['development','model_heldout']};groups['combined']=metrics(rows)
gate=json.loads((ROOT/'heldout_protocol.json').read_text())['accuracy_gate_ev'];m=groups['combined']['candidate_ev']
passed=all(r['qualified'] for r in rows) and m['mae_ev']<=gate['mae'] and m['maxabs_ev']<=gate['max_absolute_error']
report=dict(rows=rows,metrics=groups,accuracy_gate_passed=passed,scope='four exposed development states plus two model-heldout states; reference values were used in older models, not truly blind',
    weights=[.5,.5],lambda_exchange=.9,offset_ev=0,energy_labels_used_in_model=False,
    nuclear_derivatives='not qualified by this score; algebraic response modules alone do not establish a molecular Hessian',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(ROOT/'candidate_score.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
