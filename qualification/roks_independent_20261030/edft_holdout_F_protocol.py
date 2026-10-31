"""Freeze an unused-molecule test for the scoped S1 npi functional."""
from remote_bootstrap import HOST,BASE
import sys,json
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import write,digest
cases=[]
for title in ['Cyclopentadienone','Maleimide','Pyranone']:
    name=title.lower();meta=BASE/'holdout_F_inputs'/(title+'.json');geometry=BASE/'holdout_F_inputs'/(name+'.xyz')
    rows=json.loads(meta.read_text());i,r=min([(i,r) for i,r in enumerate(rows) if r['Spin']==1 and r['V/R']=='V' and r['Type']=='npi' and r.get('Safe ? (~50 meV)')=='Y'],key=lambda x:x[1]['TBE/AVTZ'])
    cases.append(dict(name=name,reference=str(BASE/'references_F'/name),metadata=str(meta),geometry=str(geometry),metadata_sha256=digest(meta),geometry_sha256=digest(geometry),
                      targets=[dict(id='n_pi',metadata_index=i,symmetry=r['State'].replace('^1','').replace('{','').replace('}','').strip(),family='npi',tbe_ev=r['TBE/AVTZ'],reference_root=1,prior_assignment_present=False)]))
protocol=dict(host=HOST,cases=cases,scope='prospective molecule holdout for S1 npi only; not promotion as a universal all-state method',
              candidate='PBE0-derived spectral GFN2 triplet plus 2*(1-.32)*K_ab, optimized with analytic orbital response',xi=.32,lambda_exchange=.9,fit=False,scalar_energy_offset=0.,
              source_sha256={n:digest(BASE/n) for n in ['edft_pbe0_response.py','edft_statefollow_response.py','edft_pbe0_statefollow.py','edft_transition_identity.py','prepare_F.py','edft_holdout_F_run.py','reference_td_checkpoint.py']},
              controls=['native ROKS','old spectral ROKS','old fixed raw/spectral 50/50 mean','parent-functional TDA on GFN2 orbitals'],existing_composite_weights=[.5,.5],
              reference='fixed first PBE0 aug-cc-pVDZ Cartesian/grid3 singlet TDA root; no substitution',minimum_projection=.7,minimum_leading_amplitude=.7,
              minimum_csf_overlap=.7,minimum_transition_overlap=.7,orbital_residual=1e-9,saddle_indices=[0,1,2,3],
              candidate_identity='overlap with prior qualified excited CSF and unmodified reference transition; native canonical CSF is diagnostic',
              gate='all three molecules qualified, candidate MAE below native, old spectral and old fixed mean, maxabs <=0.5 eV; no exclusions replaced',
              previous_general_gate='uracil pipi/all-state gate failed; remains failed. This scoped follow-up does not override it.',
              previous_E='separate preplanned E preparation is retained; triazine native seed fails transition identity. No replacement or hidden inclusion in F.',
              external_parameters='xi=.32 is the literature parameter, not fitted on MOUSE data. Independence from its original published training set is not certified.',
              source_commit='460482bcc6231dab41fe32188dc43959a7c2bdaa',nuclear_derivatives_supported=False,default_changed=False)
file=BASE/'edft_holdout_F_protocol.json'
if file.exists():assert json.loads(file.read_text())==protocol
else:write(file,protocol)
print('FROZEN F',digest(file),[c['name'] for c in cases],flush=True)
