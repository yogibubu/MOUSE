"""Remote molecular optimization of the coupled shared-pair candidate.
Reuses the converged joint electronic state and all three S0 checkpoints.
"""
import socket,sys,os,json,time,uuid,hashlib
from pathlib import Path
from dataclasses import replace
import numpy as np
BASE=Path('/home/enzo/MOUSE-roks-shared-pair-20261030');OLD=Path('/home/enzo/MOUSE-roks-local-20261030/results/formamide')
assert socket.getfqdn()=='oracle.sns.it';os.environ['MOUSE_VALIDATION_ROOT']=str(BASE)
sys.path.insert(0,str(BASE/'source/qualification'));import remote_bootstrap
from threadpoolctl import threadpool_limits
from matrix_gfn2 import GFN2Calculator,load_response_reference,reference_roks_to_native_ground,ground_referenced_roks_analytic_hessian
from matrix_gfn2.shared_pair_roks import (SharedPairROKSResult,_contexts,_SharedPairObjective,evaluate_shared_pair_roks,save_shared_pair_roks,load_shared_pair_roks)
from matrix_gfn2.ground_referenced_roks import GroundReferencedROKSResult
from matrix_gfn2.restricted_ground import RestrictedGroundResult
from matrix_gfn2.deltascf_derivatives import load_derivative_point,save_derivative_point
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from matrix_gfn2.roks_tracking import active_orbital_character
from matrix_link.optimizer import _exact_cartesian_normal_modes
ROOT=BASE/'molecular/formamide';ROOT.mkdir(parents=True,exist_ok=True)

def write(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def load(path):
 with np.load(path) as d:xyz=d['coordinates_angstrom'].copy();request=str(d['request_hash'])
 return load_derivative_point(path,request_hash=request,coordinates_angstrom=xyz,require_gradient=False)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

with threadpool_limits(limits=2):
 if (ROOT/'complete.json').exists():raise SystemExit(0)
 data=json.loads((OLD/'prepared.json').read_text());refs=[load_response_reference(p) for p in data['response_paths']]
 contract=hashlib.sha256(('shared_active_pair.v1'+''.join(r.fingerprint for r in refs)).encode()).hexdigest()
 anchors=[load(OLD/f'anchor_{i}.npz') for i in range(2)]
 calc=GFN2Calculator(data['atoms'],temperature_kelvin=0.);calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('SCC invoked'))
 accepted=ROOT/'accepted.json'
 if accepted.exists():
  manifest=json.loads(accepted.read_text());assert manifest['contract']==contract;folder=ROOT/manifest['generation']
  joint=load_shared_pair_roks(folder/'joint.npz',request_hash=contract)
  grounds=tuple(RestrictedGroundResult(load(folder/f'ground_{i}.npz'),manifest['ground_curvatures'][i]) for i in range(3));point=GroundReferencedROKSResult(joint,grounds)
  with np.load(folder/'optimizer.npz') as d:h=d['hessian'].copy();escapes=int(d['escapes'])
 else:
  manifest=json.loads((OLD/'optimization/roks_50_50_accepted/accepted.json').read_text());folder=OLD/'optimization/roks_50_50_accepted'
  parents=[load(folder/r['path']) for r in manifest['reference']];xyz=parents[0].reference_state.coordinates_bohr/ANGSTROM_TO_BOHR
  contexts=_contexts(calc,xyz,refs,parents);objective=_SharedPairObjective(contexts,parents[0].occupations)
  with np.load(BASE/'results/formamide/latest.npz') as d:coefficients=d['coefficients'].copy()
  _,gradient,curvature=objective.evaluate(coefficients);assert np.max(abs(gradient))<1e-9
  descriptors=[]
  for parent,c,ctx,ref,row in zip(parents,coefficients,contexts,refs,objective.cache['values']):
   descriptors.append(replace(parent,common_coefficients=c,fock_mixed=row['focks'][0],fock_triplet=row['focks'][1],energy_hartree=row['energy'],orbital_residual=float(np.max(abs(row['g']))),converged=False,orbital_saddle_index=None,target_orbital_saddle_index=None,gradient_hartree_per_bohr=None,reference_state=ctx.state))
  values=np.linalg.eigvalsh(curvature);seed=SharedPairROKSResult(tuple(descriptors),float(np.max(abs(gradient))),int(np.sum(values<-1e-7)),float(np.min(abs(values))),0,True)
  joint=evaluate_shared_pair_roks(calc,xyz,refs,reference_result=seed,gradient=True)
  grounds=tuple(RestrictedGroundResult(load(folder/r['path']),r['minimum_curvature_hartree']) for r in manifest['ground_references'])
  point=GroundReferencedROKSResult(joint,grounds);h=np.eye(xyz.size)*.3;escapes=0
  write(ROOT/'migration.json',dict(source_joint=str(BASE/'results/formamide/latest.npz'),sha256=sha(BASE/'results/formamide/latest.npz'),source_ground_generation=manifest['generation'] if 'generation' in manifest else manifest['reference'][0]['path'],model='unchanged parent functionals; exact sharing of the active pair',weights=[.5,.5],offset=0.))
 xyz=point.components[0].reference_state.coordinates_bohr/ANGSTROM_TO_BOHR;x=xyz.ravel()*ANGSTROM_TO_BOHR
 history=json.loads((ROOT/'trace.json').read_text()) if (ROOT/'trace.json').exists() else [];start=time.monotonic()
 def save(point,h,escapes):
  name=uuid.uuid4().hex;folder=ROOT/name;folder.mkdir()
  save_shared_pair_roks(folder/'joint.npz',point.excited,request_hash=contract)
  for i,g in enumerate(point.grounds):save_derivative_point(folder/f'ground_{i}.npz',g.reference,request_hash=contract+':ground:'+str(i),coordinates_angstrom=g.reference_state.coordinates_bohr/ANGSTROM_TO_BOHR)
  np.savez_compressed(folder/'optimizer.npz',hessian=h,escapes=escapes)
  write(accepted,dict(contract=contract,generation=name,ground_curvatures=[g.minimum_orbital_curvature_hartree for g in point.grounds]))
 def evaluate(x,previous):
  xyz=x.reshape(-1,3)/ANGSTROM_TO_BOHR
  joint=evaluate_shared_pair_roks(calc,xyz,refs,reference_result=previous.excited,gradient=True)
  for a,b,ref in zip(anchors,joint.components,refs):
   if active_orbital_character(a,b,ref.anchor_coordinates_bohr)['transition']<.7:raise ValueError('joint n-pi character is below the fixed threshold')
  result=reference_roks_to_native_ground(calc,xyz,refs,excited=joint,reference_result=previous)
  if not result.spectrum_qualified:raise ValueError('nonpositive parent gap in shared-pair surface')
  return result
 status='maximum_steps';checks=[];frequencies=None
 for iteration in range(161):
  g=point.gradient_hartree_per_bohr.ravel();energy=point.energy_hartree;maxg=float(np.max(abs(g)))
  chars=[active_orbital_character(a,b,ref.anchor_coordinates_bohr)['transition'] for a,b,ref in zip(anchors,point.components,refs)]
  row=dict(step=len(history),energy=energy,max_gradient=maxg,gaps_ev=[e*27.211386245988 for e in point.component_gaps_hartree],joint_index=point.excited.orbital_saddle_index,characters=chars)
  history.append(row);write(ROOT/'trace.json',history);save(point,h,escapes);print(json.dumps(row),flush=True)
  if maxg<3e-5:
   hh=ground_referenced_roks_analytic_hessian(calc,x.reshape(-1,3)/ANGSTROM_TO_BOHR,refs,reference_result=point);total=hh.hessian_hartree_per_bohr2
   modes=_exact_cartesian_normal_modes(data['atoms'],x.reshape(-1,3)/ANGSTROM_TO_BOHR,total);frequencies=np.asarray(modes.frequencies_cm)
   np.savez_compressed(ROOT/f'analytic_stationary_{escapes}.npz',xyz=x.reshape(-1,3)/ANGSTROM_TO_BOHR,hessian=total,gradient=g,frequencies_cm=frequencies)
   if min(frequencies)>=-20:
    direction=np.sin(np.arange(len(x))+.4);direction=direction.reshape(-1,3);direction-=direction.mean(axis=0);direction=direction.ravel()/np.linalg.norm(direction)
    for step in [2e-4,1e-4]:
     a=evaluate(x-step*direction,point);b=evaluate(x+step*direction,point)
     checks.append(dict(step=step,gradient_error=float(abs((b.energy_hartree-a.energy_hartree)/(2*step)-g@direction)),hvp_error=float(np.max(abs((b.gradient_hartree_per_bohr-a.gradient_hartree_per_bohr).ravel()/(2*step)-total@direction)))))
    status='qualified_stationary';break
   if escapes>=2:status='negative_modes_remaining';break
   delta=.05*ANGSTROM_TO_BOHR*np.asarray(modes.cartesian_directions[0]).ravel();trial=evaluate(x+delta,point)
   if trial.energy_hartree>=energy:delta=-delta;trial=evaluate(x+delta,point)
   if trial.energy_hartree>=energy:status='mode_not_downhill';break
   x=x+delta;point=trial;escapes+=1;v,u=np.linalg.eigh(total);h=(u*np.maximum(abs(v),1e-4))@u.T;continue
  v,u=np.linalg.eigh(.5*(h+h.T));delta=-u@((u.T@g)/np.maximum(v,1e-5));norm=np.linalg.norm(delta)
  if norm>.15:delta*=.15/norm
  errors=[]
  for fraction in [1.,.5,.25,.125,.0625,.03125,.015625,.0078125]:
   dx=fraction*delta
   try:
    trial=evaluate(x+dx,point)
    if trial.energy_hartree<=energy+1e-4*g@dx:break
   except Exception as exc:errors.append(str(exc))
  else:status='line_search_stalled';write(ROOT/'rejections.json',errors);break
  y=trial.gradient_hartree_per_bohr.ravel()-g;hs=h@dx;sy=dx@y;shs=dx@hs
  if shs>1e-16:
   if sy<.2*shs:theta=.8*shs/(shs-sy);y=theta*y+(1-theta)*hs;sy=dx@y
   h+=np.outer(y,y)/sy-np.outer(hs,hs)/shs
  x=x+dx;point=trial
 out=dict(molecule='formamide',status=status,maximum_gradient=float(np.max(abs(point.gradient_hartree_per_bohr))),xyz=(x.reshape(-1,3)/ANGSTROM_TO_BOHR).tolist(),checks=checks,frequencies_cm=None if frequencies is None else frequencies.tolist(),seconds=time.monotonic()-start)
 out['passed']=status=='qualified_stationary' and all(c['gradient_error']<1e-7 and c['hvp_error']<1e-6 for c in checks)
 write(ROOT/'complete.json',out);print('COMPLETE',json.dumps(out),flush=True)
