"""Freeze a genuinely unused test of the equal-energy ROKS composite."""
from remote_bootstrap import HOST,BASE
from pathlib import Path
import sys,json
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import write,digest
inputs=BASE/'holdout_D_inputs';cases=[]
for title in ['Acrolein','Pyrimidine','Nitrosomethane']:
 name=title.lower();meta=inputs/(title+'.json');geometry=inputs/(name+'.xyz');data=json.loads(meta.read_text())
 i,r=next((i,r) for i,r in enumerate(data) if r['Spin']==1 and r['V/R']=='V' and r['Type']=='npi' and r.get('Safe ? (~50 meV)')=='Y')
 cases.append(dict(name=name,reference=str(BASE/'references_D'/name),metadata=str(meta),geometry=str(geometry),metadata_sha256=digest(meta),geometry_sha256=digest(geometry),targets=[dict(id='n_pi',metadata_index=i,symmetry=r['State'].replace('^1','').replace('{','').replace('}','').strip(),family='npi',tbe_ev=r['TBE/AVTZ'],reference_root=1,prior_assignment_present=False)]))
protocol=dict(host_required='oracle.sns.it',cohort='unused molecules D: prospective validation of the fixed 50/50 energy composite',cases=cases,
 model='half stationary raw-full ROKS gap plus half stationary qualified-spectral ROKS gap; two independently optimized common-orbital singlet CSFs',weights=[.5,.5],fitted_weights=False,
 basis='aug-cc-pVDZ Cartesian',xc='PBE0',lambda_exchange=.9,gap_scale=1.,state_energy_offset=0.,
 profile_sha256=digest(BASE/'results/qualified_spectral_response/protocol.json'),profile_source_sha256=digest(BASE/'qualified_spectral_response.py'),
 calibration_source_sha256=digest(BASE/'full_fast.py'),raw_source_sha256=digest(BASE/'full_response.py'),
 reference_solver='unmodified first-singlet TDA operator; checkpointed Davidson, residual <1e-8; no root substitution',
 reference_root_rule='first singlet DFT TDA root, specified symmetry and compact projection >=0.7',
 minimum_projection=.7,minimum_leading_amplitude=.7,minimum_csf_overlap=.7,minimum_transition_overlap=.7,orbital_residual=1e-9,saddle_indices=[0,1,2,3],maximum_iterations_per_attempt=120,
 acceptance=dict(all_three_molecules_required=True,both_components_positive_and_qualified=True,ensemble_mae_below_native=True,ensemble_mae_below_both_components=True,maximum_absolute_error_ev=.5),
 profile_changes_after_predictions=False,nuclear_derivatives_supported=False,default_changed=False,source_repository='https://github.com/pfloos/QUESTDB',source_commit='460482bcc6231dab41fe32188dc43959a7c2bdaa',
 nitrosomethane_geometry='QUEST nitrosomethane_1, explicitly labelled ground state. Original missing atom-count line retained; normalized XYZ adds 6 without changing coordinates.',
 nitrosomethane_raw_geometry_sha256=digest(inputs/'nitrosomethane_1.xyz'))
p=BASE/'ensemble_holdout_D_protocol.json'
if p.exists():assert json.loads(p.read_text())==protocol
else:write(p,protocol)
print('FROZEN D',digest(p),[c['name'] for c in cases],flush=True)
