"""Couple independently assigned stationary parents without an adiabatic root detour."""
from pathlib import Path
import os,socket,sys,json,argparse,hashlib,fcntl,time
assert socket.getfqdn()=='oracle.sns.it'
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MOUSE_VALIDATION_ROOT='/home/enzo/MOUSE-orbital-performance-20261031')
R=Path('/home/enzo/MOUSE-state-following-20261031');P=Path('/home/enzo/MOUSE-publication-validation-20261030')
V=Path('/home/enzo/MOUSE-orbital-performance-20261031')
sys.path.insert(0,str(V/'source/qualification'));import remote_bootstrap
import numpy as np
from dataclasses import replace
from threadpoolctl import threadpool_limits
from matrix_gfn2 import (GFN2Calculator,FrozenNonlocalResponseReference,continue_roks_one_body_response,ROKSContinuationState)
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.deltascf_derivatives import load_derivative_point,save_derivative_point
from matrix_gfn2.roks_tracking import common_orbital_singlet_overlap,make_roks_trial_guard
from matrix_gfn2.relative_pair_roks import evaluate_relative_pair_roks,save_relative_pair_roks,load_relative_pair_roks
from matrix_gfn2.ground_referenced_roks import reference_roks_to_native_ground,GroundReferencedROKSResult
from matrix_gfn2.restricted_ground import evaluate_restricted_ground,RestrictedGroundResult

def load(p):
 with np.load(p) as d:x=d['coordinates_angstrom'].copy();req=str(d['request_hash'])
 return load_derivative_point(p,request_hash=req,coordinates_angstrom=x,require_gradient=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,default=lambda a:a.tolist() if hasattr(a,'tolist') else a.item())+'\n');temp.replace(path)
def run(name,sid):
 start=time.monotonic()
 out=V/'results'/name/sid;out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():return
 full=P/'results/full_response'/name;spectral=R/'reduced_validation/models'/name
 reports=[json.loads((V/'assigned_parents'/name/sid/profile/'majority_v1/result.json').read_text()) for profile in ['full','spectral']]
 if not all(r.get('qualified') for r in reports):raise RuntimeError('assigned stationary parents are required')
 seed=load(Path(reports[0]['checkpoint']));endpoint=load(Path(reports[1]['checkpoint']))
 s=seed.reference_state;xyz=s.coordinates_bohr/ANGSTROM_TO_BOHR
 calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
 calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Unexpected SCC'))
 ctx=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s);refs=[]
 for folder in [full,spectral]:
  path=folder/'model.npz'
  with np.load(path) as f:model={k:f[k].copy() for k in ['operators','charge','spin','linear','q0']}
  refs.append(FrozenNonlocalResponseReference.from_vertical_model(s,ctx.kernel.partition,model,sha(path)))
 case=json.loads((R/'reduced_validation/prepared'/name/'case.json').read_text());target=next(t for t in case['targets'] if t['id']==sid)
 with np.load(target['transition_file']) as f:amplitude=f['amplitude'].copy();occupied=f['occupied'].copy();virtual=f['virtual'].copy()
 # Use the original native S0 chart, never geometry-only excited metadata.
 ground=load(R/'reduced_validation/prepared'/name/'ground.npz').reference_state
 final=type('Endpoint',(),{'result':endpoint})()
 seed=replace(seed,spin_reference_fingerprint=refs[0].fingerprint)
 if (out/'stationary.npz').exists():joint=load_relative_pair_roks(out/'stationary.npz',request_hash='automatic_v2_relative',require_gradient=False)
 else:
  resume={};warm=[seed,final.result]
  if (out/'joint_latest.npz').exists():
   with np.load(out/'joint_latest.npz') as f:
    warm=[replace(v,common_coefficients=c.copy()) for v,c in zip(warm,f['coefficients'])]
    resume=dict(target_squared_overlaps=f['target_squared_overlaps'].copy(),initial_multipliers=f['multipliers'].copy())
  def joint_checkpoint(data):
   with (out/'joint_latest.tmp.npz').open('wb') as f:np.savez_compressed(f,**data)
   (out/'joint_latest.tmp.npz').replace(out/'joint_latest.npz')
  joint=evaluate_relative_pair_roks(calc,xyz,refs,initial_components=warm,anchor_components=[seed,final.result],minimum_state_overlap=.7,maximum_iterations=150,checkpoint=joint_checkpoint,**resume)
  save_relative_pair_roks(out/'stationary.npz',joint,request_hash='automatic_v2_relative')
 guards=[]
 for original,component in zip([seed,final.result],joint.components):
  guard=make_roks_trial_guard(s.overlap,component.occupations,seed.common_coefficients,reference_coefficients=ground.coefficients,occupied=occupied,virtual=virtual,reference_transition=amplitude,transition_metric='raw',minimum_transition_overlap=np.sqrt(.5))
  if not guard(original.common_coefficients,component.common_coefficients):raise RuntimeError('Joint endpoint loses fixed assigned transition: '+str(guard.last_scores))
  guards.append(guard.last_scores)
 grounds=[]
 for i,response in enumerate([None]+refs):
  folder=out/'grounds'/str(i);folder.mkdir(parents=True,exist_ok=True)
  fingerprint=None if response is None else response.fingerprint
  if (folder/'certificate.json').exists():
   cert=json.loads((folder/'certificate.json').read_text());assert cert['fingerprint']==fingerprint
   ground=RestrictedGroundResult(load(folder/'stationary.npz'),cert['minimum_curvature'],cert['index'])
  else:
   ground=evaluate_restricted_ground(calc,xyz,template=s,spin_reference=response,gradient=False)
   save_derivative_point(folder/'stationary.npz',ground.reference,request_hash='ritz_ground_checkpoint',coordinates_angstrom=xyz)
   write(folder/'certificate.json',dict(fingerprint=fingerprint,energy_hartree=ground.energy_hartree,residual=ground.orbital_residual,index=ground.orbital_saddle_index,minimum_curvature=ground.minimum_orbital_curvature_hartree))
  grounds.append(ground)
 total=GroundReferencedROKSResult(joint,tuple(grounds))
 qualified=bool(total.spectrum_qualified and joint.orbital_residual<=1e-8 and joint.minimum_absolute_orbital_curvature_hartree>=1e-7)
 write(out/'result.json',dict(host=socket.getfqdn(),molecule=name,state=sid,status='stationary_pending_assignment_audit' if qualified else 'nonconverged',excitation_ev=total.excitation_energy_hartree*27.211386245988,joint_residual=joint.orbital_residual,joint_index=joint.orbital_saddle_index,minimum_curvature_hartree=joint.minimum_absolute_orbital_curvature_hartree,guards=guards,weights=[.5,.5],lambda_exchange=.9,energy_offset_ev=0.,reference_labels_read=False,baseline_preserved=True,exploratory_model_revision='unchanged molecular reference rule; independent assigned-NTO parent initialization',parent_reports=reports,seconds_current_run=time.monotonic()-start,component_gaps_ev=[float(x*27.211386245988) for x in total.component_gaps_hartree],ground_certificates=[dict(residual=g.orbital_residual,index=g.orbital_saddle_index,minimum_curvature=g.minimum_orbital_curvature_hartree) for g in grounds]))
if __name__=='__main__':
 import traceback
 ap=argparse.ArgumentParser();ap.add_argument('molecule');ap.add_argument('state');a=ap.parse_args()
 lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:
  with threadpool_limits(limits=2):run(a.molecule,a.state)
 except Exception as exc:
  out=V/'results'/a.molecule/a.state
  write(out/'result.json',dict(host=socket.getfqdn(),molecule=a.molecule,state=a.state,status='nonconverged',reason=str(exc),traceback=traceback.format_exc(),checkpoint_preserved=True,reference_labels_read=False))
  print(traceback.format_exc(),flush=True)
