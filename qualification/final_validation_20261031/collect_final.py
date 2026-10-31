"""Portable read-only summary of all four gates; never opens sealed labels."""
from pathlib import Path
import json,socket,datetime,hashlib
assert socket.getfqdn()=='oracle.sns.it'
R=Path(__file__).resolve().parent
load=lambda p:json.loads(p.read_text()) if p.exists() else None
anchors=[r for p in sorted((R/'anchor_energy').glob('*/*/result.json')) if 'change_ev' in (r:=load(p))]
cohorts={}
for name in ['independent','independent_challenge','independent_state_challenge']:
 root=R/name;manifest=load(root/'manifest.json');pred=[]
 for row in manifest['states']:
  p=root/'predictions'/row['molecule']/row['id']/'result.json';pred.append(load(p) or dict(id=row['id'],molecule=row['molecule'],status='pending'))
 cohorts[name]=dict(manifest=manifest,predictions=pred,score=load(root/'score.json'),prediction_lock=load(root/'prediction_lock.json'),postprocess=load(root/'postprocess_progress.json'))
geometries={}
for name in ['Furan','Cyclopentadiene']:
 root=R/'geometry'/name;escape=R/'geometry_escape'/name
 geometries[name]=dict(stationary=load(root/'result.json'),mode_check=load(root/'stationary_mode_validation.json'),reference_comparison=load(root/'reference_comparison.json'),escape=load(escape/'result.json'),escape_attempt=load(escape/'attempt_closed.json'))
cost=[load(p) for p in sorted((R/'cost').glob('*/*/*/result.json'))]
report=dict(schema='mouse.final.four.gate.v1',updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=socket.getfqdn(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),anchors=anchors,baseline_definition_audit=load(R/'baseline_definition_audit.json'),cohorts=cohorts,geometries=geometries,cost=cost,preparation_audit=load(R/'preparation_cost_audit.json'),tda_validation=load(R/'true_tda_operator_validation.json'),protocol=load(R/'protocol.json'),cost_budget=load(R/'cost_budget_amendment.json'),labels_read_by_collector=False)
report['operational_closure']=dict(anchors=len(anchors)==6,independent=all(v['score'] is not None for v in cohorts.values()),geometry_attempts=all(v['escape'] is not None or v['escape_attempt'] is not None for v in geometries.values()),cost=len(cost)==72,tda_operator=report['tda_validation'] is not None)
report['scientific_scope']='Operational closure records completed attempts, not universal accuracy, representability, spectroscopic quality or an end-to-end speed advantage.'
(R/'final_results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(anchors=len(anchors),scores={k:v['score'] is not None for k,v in cohorts.items()},cost=len(cost),closure=report['operational_closure'])))
