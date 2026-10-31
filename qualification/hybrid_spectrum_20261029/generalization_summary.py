"""Summarize the frozen holdout cohort, including failures and exclusions."""
from remote_bootstrap import HOST,BASE
from pathlib import Path
import hashlib,json,shutil,sys,numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'generalization';protocol=json.loads((OUT/'protocol.json').read_text())
rows=[];all_status=[]
for case in protocol['cases']:
 name=case['name'];folder=OUT/name
 variants=[]
 for filename in ['result.json','result_index0.json']:
  if (folder/filename).exists():variants.append(json.loads((folder/filename).read_text()))
 all_status+=variants
 for t in case['targets']:
  candidates=[dict(x,variant=v.get('solver_search_scope','original positive-index search')) for v in variants for x in v['targets'] if x['id']==t['id']]
  if not candidates:
   failures=[json.loads(p.read_text()).get('error') for p in folder.glob('failure*.json')]
   rows.append(dict(molecule=name,state=t['id'],id=t['id'],roks_qualified=False,exclusion_reason='; '.join(failures) or 'not yet completed'));continue
  accepted=[x for x in candidates if x.get('roks_qualified')]
  chosen=accepted[0] if accepted else candidates[0]
  row=dict(molecule=name,**{k:v for k,v in chosen.items() if k not in ['attempts','dft_leading_configurations']})
  row['state']=row['id']
  row['solver_attempts']=[a for c in candidates for a in c['attempts']]
  if not accepted:
   row['exclusion_reason']=' | '.join(dict.fromkeys(c.get('exclusion_reason','unqualified') for c in candidates))
  rows.append(row)

def metric(values):
 a=np.array(values,float)
 return dict(count=len(a),mae_ev=float(np.mean(abs(a))) if len(a) else None,mean_signed_error_ev=float(np.mean(a)) if len(a) else None,maximum_absolute_error_ev=float(np.max(abs(a))) if len(a) else None)
qualified=[x for x in rows if x.get('roks_qualified')]
paired=[x for x in qualified if x.get('tda_qualified')]
stats=dict(roks_calibrated=metric([x['hybrid_roks_error_ev'] for x in qualified]),roks_previous_ratio=metric([x['previous_ratio_roks_error_ev'] for x in qualified]),roks_original_se=metric([x['se_roks_error_ev'] for x in qualified]),tda_calibrated=metric([x['hybrid_tda_error_ev'] for x in rows if x.get('tda_qualified')]))
stats['paired_roks_tda']=dict(count=len(paired),roks=metric([x['hybrid_roks_error_ev'] for x in paired]),tda=metric([x['hybrid_tda_error_ev'] for x in paired]))
byfamily={family:metric([x['hybrid_roks_error_ev'] for x in qualified if x['family']==family]) for family in ['npi','ppi']}
summary=dict(host=HOST,model_parameters_unchanged=True,reference_energy_scale=protocol['reference_energy_scale'],extent_power=protocol['extent_power'],planned_molecules=len(protocol['cases']),planned_states=sum(len(c['targets']) for c in protocol['cases']),qualified_states=len(qualified),excluded_states=len(rows)-len(qualified),metrics=stats,by_family=byfamily,rows=rows)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Frozen-profile generalization test','',f'Numerical work on {HOST}. Parameters fixed before this cohort: lambda0={protocol["reference_energy_scale"]:.12g}, power={protocol["extent_power"]}; no refitting.','', '**The calibrated radius-ratio profile is not generally accurate.** It substantially overestimates the small-molecule n-pi excitations, despite strong state-identity overlaps. Its paired-cohort MAE is higher than that of the original semiempirical ROKS energies. The uracil/thymine results are therefore limited evidence within a related chemical family, not a general calibration.','',f'Qualified {len(qualified)} of {summary["planned_states"]} planned states on {summary["planned_molecules"]} new molecules. Uracil and thymine are excluded from these statistics.','','| Molecule | State | QUEST / eV | New ROKS / eV | Error / eV | Previous error / eV | Index |','|---|---|---:|---:|---:|---:|---:|']
for r in rows:
 if r.get('roks_qualified'):
  attempt=next(a for a in r['solver_attempts'] if a.get('state_qualified'))
  lines.append(f'| {r["molecule"]} | {r["state"]} | {r["tbe_ev"]:.3f} | {r["hybrid_roks_ev"]:.3f} | {r["hybrid_roks_error_ev"]:+.3f} | {r["previous_ratio_roks_error_ev"]:+.3f} | {attempt["observed_index"]} |')
 else:lines.append(f'| {r["molecule"]} | {r["state"]} | {r.get("tbe_ev",float("nan")):.3f} | excluded | — | — | — |')
lines+=['','## Coverage and accuracy','',json.dumps(stats,indent=2),'','Statistics use identical accepted ROKS states for the calibrated, previous-ratio and original-SE models. Exclusions remain visible; no selection uses distance to QUEST energy.']
lines+=['','The original SE ROKS n-pi energy of HNO is 1.728469 eV versus QUEST 1.743 eV. Applying the DFT spectrum replacement raises it to 4.007790 eV. This directly identifies the spectral correction as a source of error in this case, rather than a failure to locate the correct state. The radius ratio alone does not determine how to replace an SE orbital gap while retaining the SE interaction and relaxation corrections.']
lines+=['','## Exclusions','']+[f'- {r["molecule"]} {r["state"]}: {r["exclusion_reason"]}' for r in rows if not r.get('roks_qualified')]
lines+=['','## State identity and solver scope','','Targets are linked to retained symmetry/character assignments and projected reference transition amplitudes, then checked against the relaxed GFN2 state. Canonical seed CSF overlap and reference transition overlap must both exceed 0.7; orbital chemical character must agree. Cross-basis orbital mapping uses overlap >=0.7 and margin >=0.05.','', 'The initial protocol tries electronic Hessian indices 1–3. Additional index-zero searches are recorded separately after positive-index searches lost identity or failed. Index zero is a minimum of the restricted common-orbital two-determinant CSF functional, not a claim that the excited state is a minimum of an unrestricted full electronic functional. State identity thresholds and energy-model parameters were not changed. See [Kowalczyk et al.](https://pubmed.ncbi.nlm.nih.gov/23635105/) and the [Q-Chem ROKS manual](https://manual.q-chem.com/5.3/ex_ESROKS.html).','', 'Full-functional GFN2 TDA uses the same GFN2 coefficients and the identical corrected reference spectrum. TDA values and their qualification/space limitations are retained per state in summary.json. The profile was calibrated to ROKS; these values do not establish an algorithm-neutral accuracy comparison.','', 'All ground states, unconverged latest orbitals and converged states are checkpointed. DFT references were cached PBE0/aug-cc-pVDZ calculations at the same geometries; no new DFT SCF was run.','', 'QUESTDB reference commit: 460482bcc6231dab41fe32188dc43959a7c2bdaa, CC BY-SA 4.0. Source metadata files and DFT references are hashed in each molecule result; local input copies retain the license.','']
lines+=['','## Checkpoint replay and reproduction','','All six accepted states were reloaded and their spectral corrections replayed through the molecular API, with zero difference from the recorded energies. The saved states have S(S+1)=0, orbital residuals below 1e-9 and the stated Hessian indices. Energy-only checkpoints are loaded with explicit `require_gradient=False`; the default derivative loader still requires a finite gradient. The updated regression suite passes 36 tests on Oracle.','', 'Run with the same remote interpreter and two-thread environment as the preceding campaigns:','','```','python generalization.py','python generalization.py --index-zero','python generalization_summary.py','python generalization_audit.py','```','', 'Both original positive-index attempts and additional index-zero attempts remain available, including rejected identities and failures. No model parameter was adjusted after observing this cohort.']
(OUT/'REPORT.md').write_text('\n'.join(lines))
verdict=dict(profile_id='experimental_pbe0_uracil_extent_20261029',generalization_supported=False,reason='Frozen spectral correction worsens paired-cohort MAE relative to original SE ROKS and overestimates small-molecule n-pi energies.',qualified_states=len(qualified),planned_states=summary['planned_states'],roks_calibrated_mae_ev=stats['roks_calibrated']['mae_ev'],roks_original_se_mae_ev=stats['roks_original_se']['mae_ev'],production_default_changed=False)
(OUT/'verdict.json').write_text(json.dumps(verdict,indent=2)+'\n')
inputs_dir=OUT/'inputs';inputs_dir.mkdir(exist_ok=True)
independent=Path('/home/enzo/MATRIX-runs/tda-dft-independent-20261022-v1/docs/reports/independent_dft_tda_20261022/inputs')
for name in ['Nitroxyl.json','Nitrous_acid.json','Propynal.json']:shutil.copyfile(independent/name,inputs_dir/name)
for name,folder in [('Pyridine','/home/enzo/MATRIX-runs/tda-pyridine-pyrrole-20261022-v1/docs/reports/tda_pyridine_pyrrole_20261022/pyridine/inputs'),('Cytosine','/home/enzo/MATRIX-runs/tda-frozen-validation-20261022-v1/docs/reports/tda_frozen_validation_20261022/cytosine/inputs')]:shutil.copyfile(Path(folder)/(name+'.json'),inputs_dir/(name+'.json'))
shutil.copyfile(ROOT.parent/'uracil_uridine_20261029/inputs/QUEST_LICENSE',inputs_dir/'QUEST_LICENSE')
files=[ROOT/'generalization.py',ROOT/'generalization_summary.py',ROOT/'generalization_audit.py',ROOT/'extent_support.py',OUT/'protocol.json',OUT/'index0_diagnostic.json',BASE/'source/packages/matrix-gfn2/src/matrix_gfn2/hybrid_spectrum.py',BASE/'source/packages/matrix-gfn2/src/matrix_gfn2/deltascf_derivatives.py',BASE/'source/qualification/test_energy_only_checkpoint.py']
provenance=dict(host=HOST,python=sys.version,model_profile_sha256=hashlib.sha256((ROOT/'results/extent/calibration/profile_pbe0_uracil.json').read_bytes()).hexdigest(),sources_sha256_at_collection={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},artifact_sha256={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='provenance.json' and p.suffix!='.log'})
(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))
for r in rows:print(r['molecule'],r['id'],r.get('hybrid_roks_ev'),r.get('hybrid_roks_error_ev'),r.get('exclusion_reason'))
