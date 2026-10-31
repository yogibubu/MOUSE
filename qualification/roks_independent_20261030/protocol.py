"""Freeze an untouched molecular cohort and the no-offset model definitions."""
from remote_bootstrap import HOST,BASE
from pathlib import Path
import sys,json,hashlib
OLD=Path('/home/enzo/MOUSE-roks-repair-20261030');sys.path.append(str(OLD))
from spin_kernel_probe import write,digest
ROOT=Path('/home/enzo/MATRIX-runs')
JK=ROOT/'tda-jk-qualification-20261022-v1/docs/reports/tda_jk_finalization_20261022'
IND=ROOT/'tda-dft-independent-20261022-v1/docs/reports/independent_dft_tda_20261022'
FR=ROOT/'tda-frozen-validation-20261022-v1/docs/reports/tda_frozen_validation_20261022'
cases=[]
for name,title,parent in [('cyanoformaldehyde','Cyanoformaldehyde',JK),('cyclopropenone','Cyclopropenone',JK),('methylenecyclopropene','Methylenecyclopropene',JK),('pyridazine','Pyridazine',JK),('aniline','Aniline',JK),('nitrobenzene','Nitrobenzene',JK),('benzonitrile','Benzonitrile',IND),('imidazole','Imidazole',FR/'imidazole')]:
 meta=parent/'inputs'/f'{title}.json'
 folder=(parent/'results'/f'{name}_pbe0') if name not in ['imidazole'] and parent==IND else (parent/'holdout'/f'{name}_pbe0' if parent==JK else parent/'results/pbe0_avdz')
 data=json.loads(meta.read_text());targets=[]
 for family in ['npi','ppi']:
  eligible=[(i,r) for i,r in enumerate(data) if r.get('Spin')==1 and r.get('V/R')=='V' and r.get('Type')==family and r.get('Safe ? (~50 meV)')=='Y' and r.get('TBE/AVTZ',100)<8]
  if not eligible:continue
  i,r=min(eligible,key=lambda a:a[1]['TBE/AVTZ'])
  prior=json.loads((folder/'result.json').read_text())
  assignment=next((a for a in prior.get('assignments',[]) if a['metadata_index']==i),None)
  root=assignment['reference_root'] if assignment and assignment.get('resolved') and assignment.get('family')==family else None
  targets.append(dict(reference_root=root,prior_assignment_present=assignment is not None,id='n_pi' if family=='npi' else 'pi_pi',metadata_index=i,symmetry=r['State'].replace('^1',''),tbe_ev=r['TBE/AVTZ'],family=family))
 cases.append(dict(name=name,reference=str(folder),metadata=str(meta),metadata_sha256=digest(meta),reference_sha256=digest(folder/'reference.npz'),targets=targets))
protocol=dict(host_required='oracle.sns.it',cases=cases,xc='PBE0',lambda_exchange=.9,energy_offset=False,
 models=['shell_charge_spin','tensor_spin','tensor_charge_spin'],
 tensor_definition='on-site sym(P S) sp vector and traceless pp tensor; isotropic trace average within irreducible blocks',
 calibration='DFT ground response projected onto dual moments; tensor charge is DFT minus native GFN2 charge response; no TBE fitting',
 ground_reference='minimum of the SAME corrected functional; not the native GFN2 ground energy when charge anisotropy is active',
 development_molecules=['nitroxyl','uracil','thymine','nitrous_acid','propynal','pyridine','cytosine','ketene','diazomethane','nitroaniline'],
 minimum_projection=.7,minimum_leading_amplitude=.7,minimum_csf_overlap=.7,minimum_transition_overlap=.7,
 orbital_residual=1e-9,saddle_indices=[0,1,2,3],maximum_iterations_per_attempt=120,
 selection='lowest safe valence singlet below 8eV per family; reuse independently archived resolved family/symmetry root assignment; otherwise lowest root of irrep passing projection; unresolved archived assignments are excluded; no reassignments after prediction',
 accuracy_gate=dict(minimum_new_molecules=6,minimum_npi_states=4,paired_npi_mae_strictly_lower_than_shell=True,
 paired_total_mae_not_increased=True,maximum_per_state_regression_ev=.25),
 model_changes_after_predictions=False,default_changed=False)
p=BASE/'protocol.json'
if p.exists() and 'prior_assignment_present' not in str(json.loads(p.read_text())['cases']):p.rename(BASE/'protocol_selection_draft.json')
if p.exists():assert json.loads(p.read_text())==protocol
else:write(p,protocol)
print('FROZEN',digest(p),[(c['name'],[(t['id'],t['symmetry']) for t in c['targets']]) for c in cases],flush=True)
