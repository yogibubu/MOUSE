"""Prospective spectral EDFT/RSH candidate; preserve the old 50/50 control."""
from remote_bootstrap import HOST,BASE
import sys,json
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import write,digest
cases=[]
for title in ['Diazete','Methanimine','Triazine']:
    name=title.lower();meta=BASE/'holdout_E_inputs'/(title+'.json');geometry=BASE/'holdout_E_inputs'/(name+'.xyz')
    rows=json.loads(meta.read_text());eligible=[(i,r) for i,r in enumerate(rows) if r['Spin']==1 and r['V/R']=='V' and r['Type']=='npi' and r.get('Safe ? (~50 meV)')=='Y']
    i,r=min(eligible,key=lambda t:t[1]['TBE/AVTZ'])
    cases.append(dict(name=name,reference=str(BASE/'references_E'/name),metadata=str(meta),geometry=str(geometry),metadata_sha256=digest(meta),geometry_sha256=digest(geometry),
                      targets=[dict(id='n_pi',metadata_index=i,symmetry=r['State'].replace('^1','').replace('{','').replace('}','').strip(),family='npi',tbe_ev=r['TBE/AVTZ'],reference_root=1,prior_assignment_present=False)]))
protocol=dict(host=HOST,cases=cases,selection='three unused CHN molecules, first safe valence npi state; fixed first singlet DFT reference root; no replacement after failures',
              candidate='single qualified-spectral EDFT/RSH response functional; exploratory selection after HNO/carbonyl development; not the original composite',
              existing_composite_control='old PBE0 raw/spectral energy mean, weights remain 50/50; no refitting',
              external_parameters=dict(alpha_sr=.375,alpha_lr=1.,omega=.2,xi=.32),lambda_exchange=.9,gap_scale=1.,energy_offset=0.,
              minimum_projection=.7,minimum_leading_amplitude=.7,minimum_csf_overlap=.7,minimum_transition_overlap=.7,orbital_residual=1e-9,saddle_indices=[0,1,2,3],
              source_sha256={n:digest(BASE/n) for n in ['edft_rsh_consistent_calibration.py','edft_rsh_response.py','prepare_E.py','reference_td_checkpoint.py']},
              guard_before_candidate_predictions='uracil pipi qualified and absolute error <=0.3 eV; otherwise do not run candidate holdout and reject general promotion',
              gate='all three states retained, candidate MAE below old spectral and old fixed mean and native, maxabs <=0.5 eV; no parameter tuning on E',
              preparation_allowed_before_guard=True,reference='PBE0 aug-cc-pVDZ Cartesian grid3 TDA root1 for identity only; RSH references separate for candidate',
              nuclear_derivatives_supported=False,default_changed=False,source_commit='460482bcc6231dab41fe32188dc43959a7c2bdaa')
file=BASE/'edft_holdout_E_protocol.json'
if file.exists():assert json.loads(file.read_text())==protocol
else:write(file,protocol)
print('FROZEN E',digest(file),[c['name'] for c in cases],flush=True)
