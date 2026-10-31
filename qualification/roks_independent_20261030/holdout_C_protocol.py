"""Freeze three unused npi molecules for the already fixed qualified spectral profile."""
from remote_bootstrap import HOST,BASE
from pathlib import Path
import sys,json
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import write,digest
inputs=BASE/'holdout_C_inputs';cases=[]
for title in ['Glyoxal','Pyrazine','Tetrazine']:
 name=title.lower();meta=inputs/(title+'.json');geometry=inputs/(name+'.xyz');data=json.loads(meta.read_text())
 i,r=next((i,r) for i,r in enumerate(data) if r['Spin']==1 and r['V/R']=='V' and r['Type']=='npi' and r.get('Safe ? (~50 meV)')=='Y')
 cases.append(dict(name=name,reference=str(BASE/'references_C'/name),metadata=str(meta),geometry=str(geometry),metadata_sha256=digest(meta),geometry_sha256=digest(geometry),targets=[dict(id='n_pi',metadata_index=i,symmetry=r['State'].replace('^1','').replace('{','').replace('}','').strip(),family='npi',tbe_ev=r['TBE/AVTZ'],reference_root=1,prior_assignment_present=False)]))
protocol=dict(host_required='oracle.sns.it',cohort='unused molecules C; qualified spectral profile frozen before predictions',cases=cases,
 model='qualified-spectral-full-response',basis='aug-cc-pVDZ Cartesian',xc='PBE0',lambda_exchange=.9,energy_offset=0.,
 profile_sha256=digest(BASE/'results/qualified_spectral_response/protocol.json'),profile_source_sha256=digest(BASE/'qualified_spectral_response.py'),
 calibration_source_sha256=digest(BASE/'full_fast.py'),prior_carbonyl_diagnostic='new profile worsens already exposed B carbonyls; no modification made before C',
 reference_root_rule='first singlet DFT TDA root, fixed symmetry, compact projection >=0.7, no root substitution',
 minimum_projection=.7,minimum_leading_amplitude=.7,minimum_csf_overlap=.7,minimum_transition_overlap=.7,orbital_residual=1e-9,saddle_indices=[0,1,2,3],maximum_iterations_per_attempt=120,
 acceptance=dict(all_three_molecules_required=True,mae_below_native_roks=True,mae_below_raw_full_response=True,maximum_absolute_error_ev=.5),
 profile_changes_after_predictions=False,nuclear_derivatives_supported=False,default_changed=False,source_repository='https://github.com/pfloos/QUESTDB',source_commit='460482bcc6231dab41fe32188dc43959a7c2bdaa')
p=BASE/'qualified_spectral_holdout_C_protocol.json'
if p.exists():assert json.loads(p.read_text())==protocol
else:write(p,protocol)
print('FROZEN C',digest(p),[c['name'] for c in cases],flush=True)
