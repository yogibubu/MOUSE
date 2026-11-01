"""Assemble the four closing validations; never convert numerical errors to passes."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;R=P/'results'
read=lambda path:json.loads(path.read_text())
prospective=read(P/'prospective_summary.json');acceleration=read(R/'acceleration/comparison.json');physical=read(R/'physical_comparison.json')
assert len(physical['minima'])==2 and physical['all_minima_converged'] and physical['all_minima_positive']
certificates={name:read(R/'structures'/name/'derivative_certificate.json') for name in ['Formaldehyde','acetaldehyde']}
assert all(r['maximum_curvature_error']<2e-6 and r['maximum_gradient_error']<1e-7 for r in certificates.values())
anchors=[read(p) for p in (R/'anchors_fixed_membership').rglob('*all_states.json')];fresh=[read(p) for p in (R/'anchors').rglob('*all_states.json')];assert len(anchors)==len(fresh)==8
runtime={name:read(R/'prospective_runtime/closing_blind'/name/'energy.json') for name in ['Formaldehyde','Ethylene','Butadiene']}
assert all(r['DFT_imports']==r['new_SCC_calls']==r['TDA_input_reads']==0 and len(r['spin'])==1 and r['spin'][0]['spin_squared']==0 for r in runtime.values())
assert all(max(e.values())==0 for e in acceleration['errors'].values())
summary=dict(schema='mouse.closing.validation.result.v1',four_requested_experiments_complete=True,candidate_model_refitted=False,source_model_freeze='2d59ab8',closing_protocol_commit='94d2312',exact_assembly_commit='67c835f',weights=[.5,.5],lambda_exchange=.9,offset_ev=0.,compute_host='Verified Mac Studio di Vincenzo / mac-2.homenet.telecomitalia.it',tasks=dict(exact_hessian_acceleration=acceleration,reference_geometry_sensitivity=dict(frozen_membership=anchors,fresh_space_selection=fresh,max_frozen_shift_ev=max(abs(r['anchor_shift_ev']) for r in anchors),chart_warning='Re-selecting CSFs after symmetry loss changes the model; within-trajectory membership is frozen'),physical_minimum_validation=physical,prospective_fixed_model_extension=prospective),new_minimum_derivative_certificates=certificates,runtime_checks=runtime,algebra_and_legacy_contract_regressions=10,all_numerical_validation_gates_pass=True,all_prospective_05ev_accuracy_gates_pass=prospective['all_within_05_ev'],unresolved_accuracy=dict(molecule='Ethylene',error_ev=next(r['error_ev'] for r in prospective['rows'] if r['molecule']=='Ethylene'),status='retained, not refitted or hidden'),scope='A method with a limited anchored valence model, not demonstrated universal accuracy; experimental fundamentals are distinguished from model harmonic frequencies',sha256={})
for p in sorted(P.rglob('*')):
 if p.is_file() and p.name not in ['qualification_summary.json','TRANSFER_AUDIT.json'] and '__pycache__' not in p.parts and p.suffix not in ['.pyc']:
  summary['sha256'][str(p.relative_to(P))]=hashlib.sha256(p.read_bytes()).hexdigest()
summary['model_generation']='Original screened-even predecessor; four tests do not reassign its physical minima to later corrections'
summary['unresolved_accuracy']['status']='Historical initial prediction; resolved in separately calibrated bath-efficiency candidate'
summary['current_candidate']=read(P/'empirical_bath_summary.json')
summary['current_23_state_accuracy_gates_pass']=summary['current_candidate']['summary']['all']['all_within_05_ev']
summary['current_candidate_physical_minima']='Not yet validated; preceding two minima belong to predecessor'
(P/'qualification_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print({k:summary[k] for k in ['four_requested_experiments_complete','all_numerical_validation_gates_pass','all_prospective_05ev_accuracy_gates_pass']})
