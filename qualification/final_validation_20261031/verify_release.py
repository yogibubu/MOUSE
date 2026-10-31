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
assert status['complete']==48 and status['right_censored']==0
assert d['hessian_acceleration']['qualification']['status']=='pass'
assert d['hessian_acceleration']['phthalimide_validation']['status']=='pass'
assert '46 passed' in d['hessian_acceleration']['regression_log']
for r in rows:
 if r['status']!='complete':continue
 assert r['wall_seconds']>0 and r['threads']==2 and r['host']=='oracle.sns.it'
 if r['property']=='tda':assert 'true singlet' in r['tda_operator'] and 'never DFT' in r['orbitals']
 if r['property'] in ('gradient','hessian'):
  if 'total_gradient_included' in r:assert r['total_gradient_included']
  else:
   # The three original HNO gradient records precede the explicit scope field.
   # Certify their actual executed source rather than modifying the raw rows.
   assert r['molecule']=='nitroxyl' and r['property']=='gradient' and r['repeat'] in [0,1,2]
   evidence=Path(__file__).resolve().parent/'remote_final'
   execution=json.loads((evidence/f"cost_executions/nitroxyl/gradient/{r['repeat']}.json").read_text())
   source=evidence/'source_history/partial_gradient/cost.py'
   assert hashlib.sha256(source.read_bytes()).hexdigest()==execution['script_sha256']=='30fe524444bdbaacd617785f2d423edf951dc5437bd7be25835dce967cb373f6'
   assert "gradient=point.gradient_hartree_per_bohr+gg[0]-.5*(gg[1]+gg[2])" in source.read_text()

report=dict(status='pass',data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),operational_closure=d['operational_closure'],cost_status=dict(status),scientific_claim='A completed evidence package; not a universal accuracy or end-to-end speed certificate')
(root/'qualification/final_validation_20261031/RELEASE_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
