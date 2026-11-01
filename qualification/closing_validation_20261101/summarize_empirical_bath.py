"""Aggregate every archived state; scalar JSON analysis only, no electronic job."""
from pathlib import Path
import json
P=Path(__file__).resolve().parent;refs={r['id']:r for r in json.loads((P/'downfold_diagnostic_summary.json').read_text())};rows=[];certificates=[]
for path in sorted((P/'results/bath_eta060').glob('*/*/result.json')):
 data=json.loads(path.read_text());tda_data=json.loads((path.parent/'matched_tda.json').read_text());tda={s['id']:s for s in tda_data['predictions']};assert tda_data['model']==data['model']
 runtime=json.loads((path.parent/'runtime_certificate.json').read_text());assert runtime['model']==data['model'] and max(abs(s) for s in runtime['spin_squared'])<1e-8;certificates.append(runtime)
 for s in data['states']:
  ref=refs[s['id']];t=tda[s['id']];assert t['qualified'];rows.append(dict(id=s['id'],cohort=data['cohort'],molecule=data['molecule'],family=ref['family'],prediction_ev=s['gap_ev'],reference_ev=ref['reference_ev'],error_ev=s['gap_ev']-ref['reference_ev'],overlap=s['overlap'],margin=s['margin'],model=data['model'],calibration=s['id'] in ['V010','C002'],TDA_ev=t['excitation_ev'],TDA_error_ev=t['excitation_ev']-ref['reference_ev']))
assert len(rows)==len(refs)==len({r['id'] for r in rows})==23
summary={}
for key,group in [('all',rows),('pi_pi',[r for r in rows if r['family']=='pi_pi']),('n_pi',[r for r in rows if r['family']=='n_pi']),('not_used_for_eta',[r for r in rows if not r['calibration']])]:
 summary[key]=dict(count=len(group),MAE_ev=sum(abs(r['error_ev']) for r in group)/len(group),maximum_error_ev=max(abs(r['error_ev']) for r in group),all_within_05_ev=all(abs(r['error_ev'])<.5 for r in group),TDA=dict(count=len(group),MAE_ev=sum(abs(r['TDA_error_ev']) for r in group)/len(group),maximum_error_ev=max(abs(r['TDA_error_ev']) for r in group)))
report=dict(bath_efficiency_eta=.6,eta_calibrated=True,fitted_parameters=1,calibration_states=['V010','C002'],weights=[.5,.5],lambda_exchange=.9,offset_ev=0,exposure='All 23 labels exposed development; no blind-validation claim',summary=summary,rows=sorted(rows,key=lambda r:(r['family'],r['id'])),runtime=dict(molecules=len(certificates),states=sum(len(c['spin_squared']) for c in certificates),maximum_restart_error_hartree=max(c['restart_error_hartree'] for c in certificates),DFT_imports=sum(c['DFT_imports'] for c in certificates),SCC_calls=sum(c['new_SCC_calls'] for c in certificates),TDA_solves=sum(c['TDA_solves'] for c in certificates)))
(P/'empirical_bath_summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['summary'],indent=2))
