from remote_bootstrap import HOST,BASE
from pathlib import Path
import sys,json
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import write,digest
inputs=BASE/'holdout_B_inputs';cases=[]
for title in ['Acetaldehyde','Acetone','Formamide']:
 name=title.lower();meta=inputs/(title+'.json');geometry=inputs/(name+'.xyz');data=json.loads(meta.read_text())
 i,r=next((i,r) for i,r in enumerate(data) if r['Spin']==1 and r['V/R']=='V' and r['Type']=='npi' and r.get('Safe ? (~50 meV)')=='Y')
 cases.append(dict(name=name,reference=str(BASE/'references_B'/name),metadata=str(meta),geometry=str(geometry),
  metadata_sha256=digest(meta),geometry_sha256=digest(geometry),targets=[dict(id='n_pi',metadata_index=i,symmetry=r['State'].replace('^1',''),family='npi',tbe_ev=r['TBE/AVTZ'],reference_root=1,prior_assignment_present=False)]))
protocol=dict(host_required='oracle.sns.it',cohort='fresh molecular holdout B; fixed after HNO/uracil development and before any B prediction',
 cases=cases,model='full-ao-pair-v1',basis='aug-cc-pVDZ Cartesian',xc='PBE0',lambda_exchange=.9,energy_offset=0.,
 reference_root_rule='first singlet DFT TDA root; must have specified symmetry and compact projection >=0.7; no root substitution',
 minimum_projection=.7,minimum_leading_amplitude=.7,minimum_csf_overlap=.7,minimum_transition_overlap=.7,
 orbital_residual=1e-9,saddle_indices=[0,1,2,3],maximum_iterations_per_attempt=120,
 calibration='complete DFT first and second density derivatives at embedded GFN2 ground density; no TBE fitting',
 acceptance=dict(all_three_molecules_required=True,mae_below_native_roks=True,mae_below_shell_corrected_roks=True,
 maximum_absolute_error_ev=.3),
 profile_changes_after_predictions=False,nuclear_derivatives_supported=False,default_changed=False,
 source_commit='460482bcc6231dab41fe32188dc43959a7c2bdaa',source_repository='https://github.com/pfloos/QUESTDB',
 model_source_sha256=digest(BASE/'full_fast.py'),orbital_source_sha256=digest(BASE/'full_response.py'))
p=BASE/'full_holdout_protocol.json'
if p.exists():assert json.loads(p.read_text())==protocol
else:write(p,protocol)
print('FROZEN B',digest(p),[c['name'] for c in cases],flush=True)
