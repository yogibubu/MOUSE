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
 geometries[name]=dict(stationary=load(root/'result.json'),mode_check=load(root/'stationary_mode_validation.json'),reference_comparison=load(root/'reference_comparison.json'),escape=load(escape/'result.json'),escape_attempt=load(escape/'attempt_closed.json'),minimum_mode_check=load(escape/'minimum_mode_check.json'))
cost=[load(p) for p in sorted((R/'cost').glob('*/*/*/result.json'))]
report=dict(schema='mouse.final.four.gate.v1',updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=socket.getfqdn(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),anchors=anchors,baseline_definition_audit=load(R/'baseline_definition_audit.json'),cohorts=cohorts,geometries=geometries,cost=cost,preparation_audit=load(R/'preparation_cost_audit.json'),tda_validation=load(R/'true_tda_operator_validation.json'),protocol=load(R/'protocol.json'),cost_budget=load(R/'cost_budget_amendment.json'),labels_read_by_collector=False)
report['ground_fock_candidate']={cohort:load(R/'ground_fock_candidate'/cohort/'score.json') for cohort in ['independent','independent_challenge']}
report['ground_fock_candidate_protocol']=load(R/'ground_fock_candidate_protocol.json')
report['ground_fock_prediction_lock']=load(R/'ground_fock_candidate/prediction_lock.json')
report['surface_contract_tests']=(R/'surface_contracts.log').read_text() if (R/'surface_contracts.log').exists() else None
report['csf_expansion']=[load(p) for p in sorted((R/'csf_convergence').glob('*/*/rank_*/result.json'))]
report['frame_solver_validation']=load(R/'frame_solver_validation.json')
report['solver_amendments']={p.name:load(p) for p in R.glob('*amendment.json')}
report['operational_closure']=dict(anchors=len(anchors)==6,independent=all(v['score'] is not None for v in cohorts.values()),geometry_attempts=all(v['escape'] is not None or v['escape_attempt'] is not None for v in geometries.values()),cost=len(cost)==72,tda_operator=report['tda_validation'] is not None)
report['optional_rank3_closure']=load(R/'csf_convergence/rank3_attempt_closed.json')
report['regression_log']=(R/'regression_final_solver.log').read_text() if (R/'regression_final_solver.log').exists() else None
report['benchmark_hardware']=load(R/'benchmark_hardware.json')
report['hessian_acceleration']={name:load(R/path) for name,path in {'first_complete':'hessian_acceleration/Phthalimide/result.json','qualification':'hessian_acceleration/qualification/result.json','phthalimide_validation':'hessian_acceleration/qualification/phthalimide_validation.json','benchmark_amendment':'hessian_acceleration_benchmark_amendment.json','benchmark_complete':'hessian_acceleration_benchmark_complete.json'}.items()}
report['hessian_acceleration']['tests']=(R/'pair_geometry_tests.log').read_text() if (R/'pair_geometry_tests.log').exists() else None
report['hessian_acceleration']['regression_log']=(R/'regression_hessian_acceleration.log').read_text() if (R/'regression_hessian_acceleration.log').exists() else None
report['hessian_acceleration']['surface_contracts_log']=(R/'surface_contracts_hessian_acceleration.log').read_text() if (R/'surface_contracts_hessian_acceleration.log').exists() else None
report['hessian_acceleration']['baseline_cost']=[load(p) for p in sorted((R/'cost_before_hessian_acceleration').glob('*/*/*/result.json'))]
report['closing_cost_budget']=load(R/'closing_cost_budget.json')
report['operational_closure']['phthalimide_analytic_hessian']=bool(report['hessian_acceleration']['phthalimide_validation'] and report['hessian_acceleration']['phthalimide_validation'].get('status')=='pass' and any(r['molecule']=='Phthalimide' and r['property']=='hessian' and r['repeat']==0 and r['status']=='complete' for r in cost))
report['scientific_scope']='Operational closure records completed attempts, not universal accuracy, representability, spectroscopic quality or an end-to-end speed advantage.'
(R/'final_results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(anchors=len(anchors),scores={k:v['score'] is not None for k,v in cohorts.items()},cost=len(cost),closure=report['operational_closure'])))
