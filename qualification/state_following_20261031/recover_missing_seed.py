"""Recover an excluded native checkpoint as initialization, never as a qualified result.

Run on Oracle only. Preserve the exclusion and all previous checkpoints. The
unchanged endpoint solvers must independently certify convergence and identity.
No reference excitation labels are opened by this helper.
"""
import os,sys,socket,json,hashlib,time,subprocess,shutil
from pathlib import Path
assert socket.getfqdn()=='oracle.sns.it'
R=Path('/home/enzo/MOUSE-state-following-20261031');V=Path('/home/enzo/MOUSE-orbital-performance-20261031')
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',MOUSE_VALIDATION_ROOT=str(V))
sys.path.insert(0,str(V/'source/qualification'));import remote_bootstrap
import numpy as np
from matrix_gfn2.deltascf_derivatives import load_derivative_point
from matrix_gfn2 import assigned_transition_orbitals
from matrix_gfn2.roks_tracking import make_roks_trial_guard
name='Phthalimide';sid='B024';folder=R/'reduced_validation/prepared'/name
casepath=folder/'case.json';case=json.loads(casepath.read_text());record=V/'missing_seed_recovery.json'
def load(p):
 with np.load(p) as a:x=a['coordinates_angstrom'].copy();req=str(a['request_hash'])
 return load_derivative_point(p,request_hash=req,coordinates_angstrom=x,require_gradient=False),req
if not any(t['id']==sid for t in case['targets']):
 ground,_=load(folder/'ground.npz');s=ground.reference_state
 transition=folder/sid/'transition.npz'
 with np.load(transition) as f:amp=f['amplitude'].copy();oi=f['occupied'].copy();vi=f['virtual'].copy()
 candidates=[]
 for cp in sorted((folder/sid).glob('index_*/converged.npz')):
  native,request=load(cp)
  nto=assigned_transition_orbitals(s.coefficients,s.overlap,native.occupations,oi,vi,amp)
  guard=make_roks_trial_guard(s.overlap,native.occupations,nto,reference_coefficients=s.coefficients,occupied=oi,virtual=vi,reference_transition=amp,transition_metric='rank_one',minimum_transition_overlap=np.sqrt(.5),minimum_active_overlap=0.)
  passed=guard(native.common_coefficients,native.common_coefficients)
  candidates.append(dict(checkpoint=str(cp),request_hash=request,stationary_residual=native.orbital_residual,index=native.orbital_saddle_index,passes_current_single_pair_guard=bool(passed),guard=guard.last_scores))
 usable=[r for r in candidates if r['passes_current_single_pair_guard'] and r['stationary_residual']<=1e-9]
 if not usable:raise RuntimeError('No checkpoint suitable even as an initialization seed')
 chosen=usable[0];native,_=load(Path(chosen['checkpoint']))
 target=next(t.copy() for t in json.loads((R/'reduced_scope.json').read_text())['primary_states'] if t['id']==sid)
 pair=np.unravel_index(np.argmax(abs(amp)),amp.shape);h=int(oi[pair[0]]);p=int(vi[pair[1]])
 selection=json.loads((folder/sid/'selection.json').read_text())['roots'];selected=next(r for r in selection if r['irrep_weight']>=.9 and r['projection_norm']>=.7)
 target.update(symmetry='A1',reference_root=selected['root'],prior_assignment_present=False,hole=h,particle=p,checkpoint=chosen['checkpoint'],request_hash=chosen['request_hash'],reference_transition_norm=selected['projection_norm'],index=chosen['index'],transition_file=str(transition),native_seed_only=True,native_seed_qualified_under_original_policy=False)
 report=dict(host=socket.getfqdn(),state=sid,reason='original preparation rejected absolute native CSF overlap; completed stationary point reused only as seed for independently certified corrected parents',original_exclusions=case['excluded'],case_sha256_before=hashlib.sha256(casepath.read_bytes()).hexdigest(),candidates=candidates,chosen=chosen,model_changed=False,identity_thresholds_changed=False,labels_read=False)
 backup=folder/'case_before_missing_seed_recovery.json';assert not backup.exists();shutil.copy2(casepath,backup)
 case['targets'].append(target);case['seed_recoveries']=[dict(state=sid,seed_only=True,original_exclusion_preserved=True,report=str(record))]
 record.write_text(json.dumps(report,indent=2)+'\n');casepath.write_text(json.dumps(case,indent=2)+'\n')
for profile in ['full','spectral']:
 log=V/'runs'/name/('B024_'+profile)/'recovery.log'
 (V/'progress.json').write_text(json.dumps(dict(host=socket.getfqdn(),stage='missing_seed_endpoint_recovery',state=sid,profile=profile,status='running',labels_read=False),indent=2)+'\n')
 with log.open('a') as f:
  subprocess.run([sys.executable,str(V/'repair_assigned_parent.py'),name,sid,profile,'auto','majority'],stdout=f,stderr=subprocess.STDOUT,timeout=240,check=True)
 result=json.loads((V/'assigned_parents'/name/sid/profile/'majority_v1/result.json').read_text())
 if not result.get('qualified'):raise RuntimeError(profile+' endpoint remains unresolved; see recovery log')
(V/'progress.json').write_text(json.dumps(dict(host=socket.getfqdn(),stage='missing_seed_parents_certified',state=sid,status='ready_for_coupling',labels_read=False),indent=2)+'\n')
