"""Freeze/release recovery of the identical minimal-CSF functional.
Not a production method; nuclear derivatives and transferability remain open.
"""
from pathlib import Path
import socket,os,sys,json,hashlib,time,fcntl
assert socket.getfqdn()=='oracle.sns.it'
V=Path('/home/enzo/MOUSE-orbital-performance-20261031');R=Path('/home/enzo/MOUSE-state-following-20261031');P=Path('/home/enzo/MOUSE-publication-validation-20261030');D=R/'reduced_validation';OUT=V/'pi_pi_repair'
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',MOUSE_VALIDATION_ROOT=str(V))
sys.path.insert(0,str(V/'source/qualification'));import remote_bootstrap
import numpy as np
from scipy.linalg import expm
from threadpoolctl import threadpool_limits
from matrix_gfn2 import GFN2Calculator,FrozenNonlocalResponseReference
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.deltascf_derivatives import load_derivative_point
from multicsf_objective import MultiCSFObjective
EV=27.211386245988

def write(p,value):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)

def main(name,sid,index):
 lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 out=OUT/'optimized'/name/sid/(str(index)+'_fr_v2');out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():print('existing result retained');return
 start=time.monotonic();case=json.loads((D/'prepared'/name/'case.json').read_text());target=next(t for t in case['targets'] if t['id']==sid)
 cp=D/'prepared'/name/'ground.npz'
 with np.load(cp) as a:xyz=a['coordinates_angstrom'].copy();req=str(a['request_hash'])
 s=load_derivative_point(cp,request_hash=req,coordinates_angstrom=xyz,require_gradient=False).reference_state
 calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.);native=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s)
 path=P/'results/full_response'/name/'model.npz' if index==0 else D/'models'/name/'model.npz'
 with np.load(path) as a:model={k:a[k].copy() for k in ['operators','charge','spin','q0','linear']}
 ref=FrozenNonlocalResponseReference.from_vertical_model(s,native.kernel.partition,model,hashlib.sha256(path.read_bytes()).hexdigest());ctx=prepare_gfn2_deltascf(calc,xyz,orbital_reference=s,spin_reference=ref)
 with np.load(P/'results/full_response'/name/'projected_eri.npz') as a:eri=a['eri'].copy()
 with np.load(D/'tda'/name/str(index)/'spectrum.npz') as a:c0=a['coefficients'].copy()
 with np.load(target['transition_file']) as a:amp=a['amplitude'].copy();oldoi=a['occupied'].copy();oldvi=a['virtual'].copy()
 no=len(oldoi);projected=c0[:,:no].T@s.overlap@s.coefficients[:,oldoi]@amp@s.coefficients[:,oldvi].T@s.overlap@c0[:,no:];pn=float(np.linalg.norm(projected));projected/=pn
 native_sv=np.linalg.svd(amp/np.linalg.norm(amp),compute_uv=False);rank=int(np.searchsorted(np.cumsum(native_sv**2),.95)+1)
 if rank>4:raise ValueError('predeclared four-pair diagnostic cap exceeded')
 u,sv,vh=np.linalg.svd(projected,full_matrices=True);c=np.column_stack([c0[:,:no]@u,c0[:,no:]@vh.T]);obj=MultiCSFObjective(ctx,eri,no,rank,c0,projected)
 occ=np.array([s.alpha_occupations,s.beta_occupations]);_,e0=ctx.fock_and_energy(np.array([(c0*x)@c0.T for x in occ]))
 latest=out/'latest.npz'
 if latest.exists():
  with np.load(latest) as a:assert str(a['model_fingerprint'])==ref.fingerprint;c=a['coefficients'].copy()
 from matrix_gfn2.orbital_saddle import generalized_mode_following
 history=[];evaluations=0;status='unresolved';reason='freeze-release solver did not converge'
 active=np.r_[np.arange(rank),no+np.arange(rank)]
 free=np.flatnonzero(~(np.isin(obj.left,active)|np.isin(obj.right,active)))
 def retract(point,step):
  k=np.zeros_like(point);k[obj.left,obj.right]=step;k[obj.right,obj.left]=-step;return point@expm(k)
 def guard(a,b):
  obj.evaluate(a,False);ci=obj.cache['ci'].copy();obj.evaluate(b,False);cj=obj.cache['ci'].copy();fidelity,_=obj.fidelity(b)
  sectors=min(np.min(np.linalg.norm(c0[:,:no].T@s.overlap@b[:,:rank],axis=0)),np.min(np.linalg.norm(c0[:,no:].T@s.overlap@b[:,no:no+rank],axis=0)))
  return bool(fidelity>np.sqrt(.5) and abs(obj.signed_overlap(a,ci,b,cj))>=.9 and sectors>=.7 and obj.cache['coefficient_margin']>=.05)
 def frozen_eval(point,curvature=True):
  e,g,h=obj.evaluate(point,curvature);return e,g[free],None if not curvature else h[np.ix_(free,free)]
 def frozen_retract(point,step):
  full=np.zeros(obj.dimension);full[free]=step;return retract(point,full)
 class PolishReady(Exception):pass
 polished_seed=None
 phase='frozen'
 def checkpoint(result):
  obj.evaluate(result.point,False)
  np.savez_compressed(out/('latest_'+phase+'.npz'),coefficients=result.point,ci=obj.cache['ci'],model_fingerprint=ref.fingerprint,target_index=result.target_index)
  write(out/(phase+'_history.json'),list(result.history))
  if phase=='released' and np.max(abs(result.gradient))<1e-4 and result.observed_index==result.target_index:raise PolishReady(result)
 frozen=generalized_mode_following(frozen_eval,frozen_retract,c,saddle_index=0,gradient_tolerance=1e-9,maximum_iterations=80,maximum_step=.08,trial_guard=guard,checkpoint=checkpoint)
 c=frozen.point
 if frozen.converged:
  e,g,h=obj.evaluate(c,True);target_index=int(np.count_nonzero(np.linalg.eigvalsh(h)<-1e-7));phase='released'
  try:
   released=generalized_mode_following(obj.evaluate,retract,c,saddle_index=target_index,gradient_tolerance=1e-9,maximum_iterations=100,maximum_step=.08,trial_guard=guard,checkpoint=checkpoint)
  except PolishReady as ready:
   released=ready.args[0];point=released.point.copy();polish_history=[]
   for ni in range(12):
    ee,gg,hh=obj.evaluate(point,True);polish_history.append(dict(iteration=ni,residual=float(np.max(abs(gg)))))
    if np.max(abs(gg))<=1e-9:break
    ss=np.linalg.lstsq(hh,-gg,rcond=1e-10)[0];ss*=min(1.,.08/max(np.linalg.norm(ss),1e-300));accepted=False
    for fraction in [1.,.5,.25,.125,.0625]:
     trial=retract(point,fraction*ss)
     if not guard(point,trial):continue
     et,gt,_=obj.evaluate(trial,False)
     if np.linalg.norm(gt)<np.linalg.norm(gg):point=trial;accepted=True;break
    if not accepted:break
   ee,gg,hh=obj.evaluate(point,True);vv=np.linalg.eigvalsh(hh)
   from dataclasses import replace
   released=replace(released,point=point,energy=ee,gradient=gg,eigenvalues=vv,observed_index=int(np.count_nonzero(vv<-1e-7)),converged=bool(np.max(abs(gg))<=1e-9 and np.count_nonzero(vv<-1e-7)==target_index and np.min(abs(vv))>1e-7))
   np.savez_compressed(out/'latest_released.npz',coefficients=point,ci=obj.cache['ci'],model_fingerprint=ref.fingerprint,target_index=target_index)
   write(out/'newton_polish_history.json',polish_history)
  c=released.point;history=list(released.history);evaluations=released.iterations
  if released.converged:status='stationary_candidate'
  else:reason='released saddle search unresolved; checkpoint retained'
 else:reason='frozen spectator relaxation unresolved; checkpoint retained'
 e,g,h=obj.evaluate(c,True);values=np.linalg.eigvalsh(h);fidelity,raw=obj.fidelity(c)
 result=dict(host=socket.getfqdn(),molecule=name,state=sid,profile=index,status=status,reason=None if status=='stationary_candidate' else reason,rank=rank,rank_selection='minimum rank retaining 95 percent of original fixed projected transition; no energy used',retained_weight=float(np.sum(native_sv[:rank]**2)),excitation_ev=float((e-e0)*EV),residual=float(np.max(abs(g))),orbital_index=int(np.count_nonzero(values< -1e-7)),minimum_absolute_curvature=float(np.min(abs(values))),fidelity=fidelity,raw_transition=raw,projection_norm=pn*target['reference_transition_norm'],ci=obj.cache['ci'].tolist(),seconds=time.monotonic()-start,evaluations=evaluations,model_fingerprint=ref.fingerprint,offdiagonal='unscaled bare projected 2(ia|jb)-(ij|ab); different multistate functional, not a changed TDA kernel',diagonal_lambda_exchange=.9,energy_offset_ev=0.,weights=[.5,.5],nuclear_derivatives_available=False,accuracy_labels_already_exposed=True)
 write(out/'result.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=2):main(sys.argv[1],sys.argv[2],int(sys.argv[3]))
