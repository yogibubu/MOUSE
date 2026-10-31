"""Deterministic assigned-NTO initialization of the unchanged endpoint model."""
from pathlib import Path
import os,sys,socket,json,hashlib,time,fcntl
assert socket.getfqdn()=='oracle.sns.it'
R=Path('/home/enzo/MOUSE-state-following-20261031')
V=Path('/home/enzo/MOUSE-orbital-performance-20261031')
P=Path('/home/enzo/MOUSE-publication-validation-20261030')
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',MOUSE_VALIDATION_ROOT=str(V))
sys.path.insert(0,str(V/'source/qualification'));import remote_bootstrap
import numpy as np
from dataclasses import replace
from threadpoolctl import threadpool_limits
from matrix_gfn2 import GFN2Calculator,FrozenNonlocalResponseReference
from matrix_gfn2.roks import evaluate_gfn2_roks_singlet
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.deltascf_derivatives import load_derivative_point,save_derivative_point
from matrix_gfn2.roks_tracking import make_roks_trial_guard

def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);t=path.with_suffix('.tmp')
 t.write_text(json.dumps(value,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else x.item())+'\n');t.replace(path)
def load(path):
 with np.load(path) as f:xyz=f['coordinates_angstrom'].copy();req=str(f['request_hash'])
 return load_derivative_point(path,request_hash=req,coordinates_angstrom=xyz,require_gradient=False),xyz

def run(name,sid,profile,index=None,search=False,majority=False,native_only=False):
 out=V/'assigned_parents'/name/sid/profile
 if majority:out=out/'majority_v1'
 if search:out=out/'initialization_search'/str(index)
 elif index is not None:out=out/'indexed'/str(index)
 out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists() and json.loads((out/'result.json').read_text()).get('qualified'):return
 case=json.loads((R/'reduced_validation/prepared'/name/'case.json').read_text());target=next(t for t in case['targets'] if t['id']==sid)
 native,xyz=load(Path(target['checkpoint']));ground,_=load(R/'reduced_validation/prepared'/name/'ground.npz');s=ground.reference_state
 calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
 ctx=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s)
 path=(P/'results/full_response'/name/'model.npz' if profile=='full' else R/'reduced_validation/models'/name/'model.npz')
 with np.load(path) as f:model={k:f[k].copy() for k in ['operators','charge','spin','q0','linear']}
 ref=FrozenNonlocalResponseReference.from_vertical_model(s,ctx.kernel.partition,model,hashlib.sha256(path.read_bytes()).hexdigest())
 with np.load(target['transition_file']) as f:amp=f['amplitude'].copy();oi=f['occupied'].copy();vi=f['virtual'].copy()
 from matrix_gfn2 import assigned_transition_orbitals
 nto=assigned_transition_orbitals(s.coefficients,s.overlap,native.occupations,oi,vi,amp)
 candidates=[('assigned_nto',nto)]
 if search or majority:candidates.append(('native_stationary',native.common_coefficients))
 if native_only:candidates=[('native_stationary',native.common_coefficients)]
 if profile=='full' and not native_only:
  old=P/'results/full_response'/name/sid/'result.json'
  if old.exists():
   report=json.loads(old.read_text())
   if report.get('qualified'):
    original,_=load(Path(report['checkpoint']));candidates.insert(0,('retained_full',original.common_coefficients))
 attempts=[];start=time.monotonic()
 for solver in (['newton','gmf'] if index is None else ['gmf']):
  for label,c in candidates:
   seed=replace(native,common_coefficients=c,reference_state=s,spin_reference_fingerprint=ref.fingerprint,spin_frame_clusters=getattr(ref,'frame_clusters',None),converged=False,orbital_saddle_index=None,target_orbital_saddle_index=None)
   guard=make_roks_trial_guard(s.overlap,native.occupations,nto,reference_coefficients=s.coefficients,occupied=oi,virtual=vi,reference_transition=amp,transition_metric='rank_one',minimum_transition_overlap=np.sqrt(.5) if majority else .9,minimum_active_overlap=0. if majority else .7)
   final_guard=guard
   if search:guard=make_roks_trial_guard(s.overlap,native.occupations,nto,reference_coefficients=s.coefficients,occupied=oi,virtual=vi,reference_transition=amp,transition_metric='rank_one',minimum_transition_overlap=0.)
   if not guard(c,c):
    attempts.append(dict(seed=label,solver=solver,status='initial_identity_rejected',guard=guard.last_scores));continue
   begin=time.monotonic()
   try:
    options=(dict(maximum_evaluations=120,maximum_step=.08,certify_index=True) if solver=='newton' else dict(saddle_index=index,freeze_and_release=True,freeze_iterations=100,maximum_iterations=150,maximum_step=.08))
    result=evaluate_gfn2_roks_singlet(calc,xyz,*native.occupations,reference_result=seed,spin_reference=ref,reference_policy='geometry_only',minimum_state_overlap=.7,orbital_solver=solver,orbital_tolerance=1e-9,trial_guard=guard,**options)
    if not final_guard(result.common_coefficients,result.common_coefficients):raise RuntimeError('final assigned transition rejected: '+str(final_guard.last_scores))
    save_derivative_point(out/'stationary.npz',result,request_hash='assigned_nto_parent_v1',coordinates_angstrom=xyz)
    write(out/'result.json',dict(host=socket.getfqdn(),molecule=name,state=sid,profile=profile,qualified=True,solver=solver,seed=label,residual=result.orbital_residual,index=result.orbital_saddle_index,energy_hartree=result.energy_hartree,guard=final_guard.last_scores,initialization_search=search,identity_policy='attainable_single_pair_fidelity_gt_sqrt_half_plus_margin' if majority else 'rank_one_fidelity_0.90',checkpoint=str(out/'stationary.npz'),request_hash='assigned_nto_parent_v1',seconds=time.monotonic()-start,attempts=attempts,model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),reference_labels_read=False))
    print((out/'result.json').read_text(),flush=True);return
   except (ValueError,RuntimeError) as exc:
    row=dict(seed=label,solver=solver,status='unresolved',reason=str(exc),guard=guard.last_scores,seconds=time.monotonic()-begin)
    attempts.append(row);write(out/'attempts.json',attempts);print(row,flush=True)
 write(out/'result.json',dict(host=socket.getfqdn(),molecule=name,state=sid,profile=profile,qualified=False,attempts=attempts,seconds=time.monotonic()-start,reference_labels_read=False))

if __name__=='__main__':
 lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 with threadpool_limits(limits=2):run(*sys.argv[1:4],index=(None if sys.argv[4]=='auto' else int(sys.argv[4])) if len(sys.argv)>4 else None,search='search' in sys.argv[5:],majority='majority' in sys.argv[5:],native_only='native' in sys.argv[5:])
