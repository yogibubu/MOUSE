"""Lightweight final evidence checks; no electronic calculations."""
from pathlib import Path
import json, math, hashlib, collections
root=Path(__file__).resolve().parents[2]
data=root/'manuscripts/mouse_orbital_optimized_dft/data/final_validation.json'
d=json.loads(data.read_text())
assert all(d['operational_closure'].values()), d['operational_closure']
assert len(d['anchors'])==6 and all(math.isfinite(r['change_ev']) for r in d['anchors'])
assert d['cohorts']['independent']['score']['n_attempted']==6
assert d['cohorts']['independent']['score']['n_matched']==4
assert d['cohorts']['independent_challenge']['score']['n_matched']==2
assert d['cohorts']['independent_state_challenge']['score']['n_matched']==3
assert '46 passed' in d['regression_log']
assert '2 passed' in d['surface_contract_tests']
assert d['geometries']['Cyclopentadiene']['escape']['status']=='minimum_certified'
assert min(d['geometries']['Cyclopentadiene']['escape']['frequencies_cm1'])>0
assert d['geometries']['Cyclopentadiene']['minimum_mode_check']['status']=='pass'
assert not d['geometries']['Furan']['escape_attempt']['nuclear_minimum']
rows=d['cost']; assert len(rows)==72
assert len({(r['molecule'],r['property'],r['repeat']) for r in rows})==72
status=collections.Counter(r['status'] for r in rows)
assert status['not_repeated_due_to_prespecified_budget']==24
assert status['complete']+status['right_censored']==48
for r in rows:
 if r['status']!='complete':continue
 assert r['wall_seconds']>0 and r['threads']==2 and r['host']=='oracle.sns.it'
 if r['property']=='tda':assert 'true singlet' in r['tda_operator'] and 'never DFT' in r['orbitals']
 if r['property'] in ('gradient','hessian'):assert r['total_gradient_included']
report=dict(status='pass',data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),operational_closure=d['operational_closure'],cost_status=dict(status),scientific_claim='A completed evidence package; not a universal accuracy or end-to-end speed certificate')
(root/'qualification/final_validation_20261031/RELEASE_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
