"""Oracle-only exploratory state-following recovery, unchanged endpoint energy.
No scoring labels are read. Baseline failures and predictions remain immutable.
"""
from pathlib import Path
import socket,os,sys,json,hashlib,time,argparse,traceback
assert socket.getfqdn()=='oracle.sns.it'
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MOUSE_VALIDATION_ROOT='/home/enzo/MOUSE-state-following-20261031')
R=Path('/home/enzo/MOUSE-state-following-20261031');P=Path('/home/enzo/MOUSE-publication-validation-20261030')
sys.path.insert(0,str(R/'source/qualification'));import remote_bootstrap
import numpy as np
from dataclasses import replace
from threadpoolctl import threadpool_limits
from matrix_gfn2 import GFN2Calculator,FrozenNonlocalResponseReference,evaluate_gfn2_roks_singlet
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.deltascf_derivatives import load_derivative_point,save_derivative_point
from matrix_gfn2.roks_tracking import make_roks_trial_guard,common_orbital_singlet_overlap
from matrix_gfn2.relative_pair_roks import evaluate_relative_pair_roks,save_relative_pair_roks
from matrix_gfn2.ground_referenced_roks import reference_roks_to_native_ground

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,default=lambda a:a.tolist() if hasattr(a,'tolist') else a.item())+'\n');t.replace(p)
def load(path,request=None):
 with np.load(path) as d:xyz=d['coordinates_angstrom'].copy();request=request or str(d['request_hash'])
 return load_derivative_point(path,request_hash=request,coordinates_angstrom=xyz,require_gradient=False)
def run(name,sid):
 out=R/'recovery'/name/sid;out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():return
 case=json.loads((P/'blind/prepared'/name/'case.json').read_text());target=next(t for t in case['targets'] if t['id']==sid)
 full=P/'results/full_response'/name;spectral=P/'results/qualified_spectral_response'/name
 report=json.loads((full/sid/'result.json').read_text());assert report['qualified']
 parent=load(Path(report['checkpoint']),report['request_hash']);s=parent.reference_state;xyz=s.coordinates_bohr/ANGSTROM_TO_BOHR
 atoms=[{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers];calc=GFN2Calculator(atoms,temperature_kelvin=0.)
 calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Unexpected SCC'))
 native=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s)
 models=[]
 for f in [full/'model.npz',spectral/'model.npz']:
  with np.load(f) as d:models.append({k:d[k].copy() for k in ['operators','charge','spin','q0','linear']})
 for k in ['operators','charge','spin','q0']:assert np.max(abs(models[0][k]-models[1][k]))<1e-12
 with np.load(target['transition_file']) as d:amplitude=d['amplitude'].copy();occupied=d['occupied'].copy();virtual=d['virtual'].copy()
 def response(t):
  data=dict(models[0]);data['linear']=(1-t)*models[0]['linear']+t*models[1]['linear']
  return FrozenNonlocalResponseReference.from_vertical_model(s,native.kernel.partition,data,'continuation/'+sha(full/'model.npz')+'/'+sha(spectral/'model.npz')+'/'+float(t).hex())
 ref0=response(0.);parent=replace(parent,spin_reference_fingerprint=ref0.fingerprint)
 ctx=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s,spin_reference=ref0)
 f,e=ctx.fock_and_energy(parent.densities());ft,et=ctx.fock_and_energy(parent.densities(triplet=True));assert abs(2*e-et-parent.energy_hartree)<1e-9
 anchor=parent;current=parent;t=0.;dt=.2;history=[];start=time.perf_counter()
 if (out/'accepted.json').exists():
  saved=json.loads((out/'accepted.json').read_text());t=saved['fraction'];current=load(out/saved['checkpoint']);history=json.loads((out/'history.json').read_text())
 while t<1-1e-12:
  trial_t=min(1.,t+dt);ref=response(trial_t)
  guard=make_roks_trial_guard(s.overlap,current.occupations,anchor.common_coefficients,minimum_local_overlap=.90,minimum_active_overlap=.70,reference_coefficients=s.coefficients,occupied=occupied,virtual=virtual,reference_transition=amplitude,minimum_transition_overlap=.70,minimum_sector_norm=.70)
  if not guard(current.common_coefficients,current.common_coefficients):raise RuntimeError('Assigned full-response seed fails fixed transition/active gates: '+str(guard.last_scores))
  seed=replace(current,spin_reference_fingerprint=ref.fingerprint,converged=False,orbital_saddle_index=None,target_orbital_saddle_index=None)
  try:
   result=evaluate_gfn2_roks_singlet(calc,xyz,*current.occupations,reference_result=seed,spin_reference=ref,reference_policy='geometry_only',minimum_state_overlap=.90,orbital_solver='newton',orbital_tolerance=1e-9,maximum_evaluations=80,maximum_step=.08,certify_index=True,trial_guard=guard)
   if not guard(current.common_coefficients,result.common_coefficients):raise RuntimeError('Converged trial fails transition guard')
   local=common_orbital_singlet_overlap(current.common_coefficients,result.common_coefficients,s.overlap,current.occupations)
   row=dict(from_fraction=t,to_fraction=trial_t,accepted=True,index=result.orbital_saddle_index,residual=result.orbital_residual,local_csf=local,guard=guard.last_scores,guard_trials=guard.evaluations,guard_rejections=int(guard.rejections))
   t=trial_t;current=result;cp=f'accepted_{len(history):04d}.npz';req='state_continuation_v1:'+float(t).hex();save_derivative_point(out/cp,current,request_hash=req,coordinates_angstrom=xyz)
   write(out/'accepted.json',dict(fraction=t,checkpoint=cp,request_hash=req));dt=min(.2,dt*1.5)
  except Exception as exc:
   row=dict(from_fraction=t,to_fraction=trial_t,accepted=False,error=str(exc),guard=guard.last_scores,guard_trials=guard.evaluations,guard_rejections=int(guard.rejections));dt*=.5
   if dt<.0005:
    history.append(row);write(out/'history.json',history);raise RuntimeError('Continuity-preserving branch cannot be followed to endpoint: '+str(row)) from exc
  history.append(row);write(out/'history.json',history);print(name,sid,row,flush=True)
 ref1=response(1.);ctx=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s,spin_reference=ref1)
 f,e=ctx.fock_and_energy(current.densities());ft,et=ctx.fock_and_energy(current.densities(triplet=True));endpoint_error=abs(2*e-et-current.energy_hartree);assert endpoint_error<1e-10
 joint=evaluate_relative_pair_roks(calc,xyz,[ref0,ref1],initial_components=[anchor,current],minimum_state_overlap=.7)
 save_relative_pair_roks(out/'stationary.npz',joint,request_hash='recovered_relative_v1')
 total=reference_roks_to_native_ground(calc,xyz,[ref0,ref1],excited=joint,gradient=False)
 scoreguard=make_roks_trial_guard(s.overlap,current.occupations,anchor.common_coefficients,reference_coefficients=s.coefficients,occupied=occupied,virtual=virtual,reference_transition=amplitude)
 assert scoreguard(current.common_coefficients,current.common_coefficients)
 result=dict(host=socket.getfqdn(),molecule=name,state=sid,status='matched' if total.spectrum_qualified else 'nonpositive_gap',excitation_ev=total.excitation_energy_hartree*27.211386245988,endpoint_energy_check_hartree=endpoint_error,joint_residual=joint.orbital_residual,joint_index=joint.orbital_saddle_index,minimum_curvature=joint.minimum_absolute_orbital_curvature_hartree,final_guard=scoreguard.last_scores,seconds_current_run=time.perf_counter()-start,weights=[.5,.5],lambda_exchange=.9,energy_offset=0.,baseline_untouched=True,exploratory_solver_recovery=True,reference_labels_read=False,baseline_result=json.loads((P/'blind/mouse'/name/'result.json').read_text()) if (P/'blind/mouse'/name/'result.json').exists() else None)
 write(out/'result.json',result);print('RECOVERED',name,sid,result,flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--molecule',default='Cyclopentadiene');ap.add_argument('--state',default='B016');a=ap.parse_args()
 protocol=dict(host=socket.getfqdn(),method='adaptive F-to-Q continuation with a local CSF and fixed global transition trial guard; certified analytic orbital Newton; final relative-pair solve unchanged',source_sha256={str(p):sha(p) for p in [Path(__file__),R/'source/packages/matrix-gfn2/src/matrix_gfn2/orbital_saddle.py',R/'source/packages/matrix-gfn2/src/matrix_gfn2/roks_tracking.py',R/'source/packages/matrix-gfn2/src/matrix_gfn2/roks_orbital_response.py']},weights=[.5,.5],lambda_exchange=.9,energy_offset_ev=0.,guard_thresholds=dict(local_csf=.9,active=.7,transition=.7,sector_norm=.7),initial_step=.2,minimum_step=.0005,maximum_step=.2,newton_iterations=80,fit_reference_energies=False,exploratory_numerical_recovery_not_original_blind_baseline=True)
 f=R/'protocol.json'
 if f.exists():assert json.loads(f.read_text())==protocol
 else:write(f,protocol)
 with threadpool_limits(limits=2):run(a.molecule,a.state)
