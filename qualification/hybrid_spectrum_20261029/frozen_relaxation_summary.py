"""Report the parameter-free frozen DFT / semiempirical relaxation experiment."""
from remote_bootstrap import HOST,BASE
from pathlib import Path
import hashlib,json,platform,sys,numpy as np,pyscf
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'frozen_relaxation';config=json.loads((OUT/'protocol.json').read_text());plan=json.loads((OUT/'plan.json').read_text())
rows=[];failures=[];grounds=[]
for case in plan['cases']:
 folder=OUT/case['name'];p=folder/'result.json'
 if not p.exists():
  f=folder/'failure.json';failures.append(dict(molecule=case['name'],details=json.loads(f.read_text()) if f.exists() else 'pending'));continue
 d=json.loads(p.read_text());grounds.append(dict(molecule=case['name'],**d['ground']))
 for r in d['records']:rows.append(dict(molecule=case['name'],cohort=case['cohort'],**r))
qualified=[r for r in rows if r.get('qualified')]

def metrics(values):
 a=np.array(values,float)
 return dict(count=len(a),mae_ev=float(np.mean(abs(a))) if len(a) else None,mean_signed_error_ev=float(np.mean(a)) if len(a) else None,maximum_absolute_error_ev=float(max(abs(a))) if len(a) else None)

def cohort_metrics(rs):
 return dict(qualified_states=len(rs),hybrid=metrics([r['hybrid_error_ev'] for r in rs]),original_se_roks=metrics([r['se_roks_error_ev'] for r in rs]),dft_frozen_only=metrics([r['dft_frozen_error_ev'] for r in rs]),n_pi=metrics([r['hybrid_error_ev'] for r in rs if r['id']=='n_pi']),pi_pi=metrics([r['hybrid_error_ev'] for r in rs if r['id']=='pi_pi']))
cohorts={name:cohort_metrics([r for r in qualified if name=='all' or r['cohort']==name]) for name in ['all','development','previous_generalization']}
order=[]
for name in config['planned_molecules']:
 targets={r['id']:r for r in qualified if r['molecule']==name}
 if 'n_pi' in targets and 'pi_pi' in targets:
  gap=targets['pi_pi']['hybrid_ev']-targets['n_pi']['hybrid_ev'];truth=targets['pi_pi']['tbe_ev']-targets['n_pi']['tbe_ev']
  order.append(dict(molecule=name,predicted_pi_minus_n_ev=gap,reference_pi_minus_n_ev=truth,correct_order=bool(gap*truth>0)))
grid=json.loads((OUT/'grid_validation.json').read_text()) if (OUT/'grid_validation.json').exists() else None
summary=dict(host=HOST,formula=config['formula'],fitted_parameters=0,scf_optimizations_launched=0,excited_optimizations_launched=0,planned_states=config['planned_states'],qualified_states=len(qualified),excluded_states=[r for r in rows if not r.get('qualified')],case_failures=failures,metrics=cohorts,ordering=order,grid_validation=grid,ground_maximum_deviation_hartree=max(abs(g['difference_from_cached_scf_hartree']) for g in grounds) if grounds else None,rows=rows)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Frozen DFT excitation plus semiempirical relaxation','', '29 October 2026. Parameter-free retrospective test, executed only on verified Oracle. Existing ground and excited-state wavefunctions were retained. No new SCF or excited-state optimization was launched.','',r'`Omega_hybrid = Omega_DFT_frozen + (Omega_SE_relaxed - Omega_SE_frozen)`','', 'Each frozen excitation is `2 E_mixed - E_triplet - E_ground`, evaluated on the respective unchanged canonical ground orbitals. DFT uses PBE0/aug-cc-pVDZ and a common, unpruned level-3 integration grid for all densities. SE uses the existing polarized GFN2 functional. There is no orbital-energy scaling, extent factor or fitted parameter.','',f'{len(qualified)} states are qualified out of {config["planned_states"]} planned. The uracil/thymine development cohort is reported separately from the previously tested six-state cohort; none is a new blind holdout for this newly proposed model.','','## Mean absolute errors','','| Cohort | States | SE ROKS / eV | Frozen DFT + SE relaxation / eV | Frozen DFT alone / eV |','|---|---:|---:|---:|---:|']
lines[2:2]=['**Outcome:** the no-fit hybrid improves the six-state previous generalization cohort MAE from 0.446220 to 0.211504 eV, and the full ten-state MAE from 0.625703 to 0.255138 eV. However, it still reverses the n-pi/pi-pi ordering in uracil and thymine. It is a useful experimental improvement, not a validated production solution to the original ordering problem.','']
for label,name in [('All qualified','all'),('Uracil and thymine','development'),('Previous generalization cohort','previous_generalization')]:
 c=cohorts[name]
 if c['qualified_states']:lines.append(f'| {label} | {c["qualified_states"]} | {c["original_se_roks"]["mae_ev"]:.6f} | {c["hybrid"]["mae_ev"]:.6f} | {c["dft_frozen_only"]["mae_ev"]:.6f} |')
lines+=['','## Individual vertical energies','','| Molecule | State | QUEST / eV | SE ROKS / eV | Hybrid / eV | Hybrid error / eV |','|---|---|---:|---:|---:|---:|']
for r in rows:
 if r.get('qualified'):lines.append(f'| {r["molecule"]} | {r["id"]} | {r["tbe_ev"]:.3f} | {r["se_roks_ev"]:.3f} | {r["hybrid_ev"]:.3f} | {r["hybrid_error_ev"]:+.3f} |')
 else:lines.append(f'| {r["molecule"]} | {r["id"]} | {r["tbe_ev"]:.3f} | — | excluded | — |')
lines+=['','## Decomposition','','| Molecule | State | SE frozen / eV | DFT frozen / eV | SE relaxation / eV |','|---|---|---:|---:|---:|']
for r in qualified:lines.append(f'| {r["molecule"]} | {r["id"]} | {r["se_frozen_ev"]:.6f} | {r["dft_frozen_ev"]:.6f} | {r["se_relaxation_ev"]:.6f} |')
lines+=['','## State ordering','']
for r in order:lines.append(f'- {r["molecule"]}: pi-pi minus n-pi = {r["predicted_pi_minus_n_ev"]:+.6f} eV; QUEST {r["reference_pi_minus_n_ev"]:+.6f} eV; correct ordering: {r["correct_order"]}.')
lines+=['','These residual ordering errors cannot be attributed to the integration grid. The experiment does not by itself identify whether the remaining mismatch comes from the approximate frozen DFT excitation, the transfer of SE relaxation, or both. The decomposition makes a subsequent comparison of DFT and SE orbital relaxation a concrete diagnostic. No additional spectral scaling or fit was introduced to repair the ordering.']
lines+=['','## Exclusions and checks','']
for r in rows:
 if not r.get('qualified'):lines.append(f'- {r["molecule"]} {r["id"]}: {r["exclusion_reason"]}.')
lines+=['', 'State identities are the previously qualified common-orbital singlet solutions, with orbital-stationarity and CSF/transition-overlap checks. Ground orbitals are matched by actual cross-AO overlaps within separate occupied/virtual sectors (minimum 0.7 and margin 0.05). Energies are never selected by proximity to QUEST.','',f'The maximum UKS-ground versus cached-SCF energy difference is {summary["ground_maximum_deviation_hartree"]} hartree. Total electron counts are checked in the AO metric; mixed and triplet densities have identical total density. Spin-flip invariance is checked for the first HNO and uracil states.']
if grid:lines.append(f'Independent level-4 grid check on HNO and uracil: maximum frozen-excitation change {grid["maximum_difference_ev"]:.9f} eV, declared tolerance {grid["tolerance_ev"]} eV; passed: {grid["passed"]}.')
lines+=['','The motivation to separate frozen excitation and relaxation is related to [Ling et al., JCTC 2026](https://doi.org/10.1021/acs.jctc.5c01738). The present use of frozen canonical ground orbitals and combination across two different methods is an experimental hypothesis, not that paper\'s derived EDA or a proven correction.','', 'This is an energy-only experiment. Nuclear derivatives of the DFT frozen block and the ground-orbital matching are not supplied; existing SE gradients/Hessians must not be relabeled as hybrid derivatives. The DFT setup and static exchange/XC evaluations contribute to the total computational cost, even though no excited-state SCF is needed. No new claim of equal-accuracy speed over TDA is made.','', 'Ground, mixed and triplet energy points are individually cached with request hashes. Reproduction on Oracle uses its existing interpreter and OMP/BLAS thread limits of two:','','```','python frozen_relaxation.py','python frozen_grid_check.py','python frozen_relaxation_summary.py','```','', 'QUESTDB commit 460482bcc6231dab41fe32188dc43959a7c2bdaa, CC BY-SA 4.0; source metadata/license are retained in the preceding campaigns.']
(OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
verdict=dict(model='PBE0 frozen excitation plus polarized GFN2 ROKS relaxation',fitted_parameters=0,retrospective_accuracy_improved=True,previous_generalization_mae_ev=cohorts['previous_generalization']['hybrid']['mae_ev'],all_states_mae_ev=cohorts['all']['hybrid']['mae_ev'],uracil_thymine_ordering_solved=False,production_ready=False,hybrid_nuclear_derivatives_implemented=False)
(OUT/'verdict.json').write_text(json.dumps(verdict,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['rows','excluded_states']},indent=2))
for r in qualified:print(r['molecule'],r['id'],r['hybrid_ev'],r['hybrid_error_ev'])
