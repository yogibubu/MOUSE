"""Continue a recalibrated anchor without changing its frozen energy manifold."""
from common import *
from matrix_gfn2.multicsf_frame import RelativeActiveFrame
import fcntl,argparse

def run(name,sid,scale):
 import anchor_energy as original
 base=MultiCSFModel(W/f'models/{name}/{sid}/model.json');out=OUT/f'anchor_energy/{name}/{scale:.2f}'
 if (out/'result.json').exists():return json.loads((out/'result.json').read_text())
 model=MultiCSFModel(out/'portable_model/model.json');calc=calculator(model.anchor);objects,point=model.objectives(calc,base.xyz);frame=RelativeActiveFrame(objects);cp=out/'frame_continuation.npz';history=[]
 if cp.exists():
  with np.load(cp) as a:assert str(a['model_fingerprint'])==model.fingerprint;point=list(a['coefficients'].copy())
 if (out/'frame_continuation_history.json').exists():history=json.loads((out/'frame_continuation_history.json').read_text())
 target=point[0][:,frame.active].T@frame.s@point[1][:,frame.active]
 for iteration in range(96):
  e,g,h,t,rawmap,v=frame.evaluate(point);residual=float(np.max(abs(g)));history.append(dict(iteration=len(history),energy=e,residual=residual,fidelity=[o.fidelity(c)[0] for o,c in zip(objects,point)]));write(out/'frame_continuation_history.json',history)
  temporary=cp.with_suffix('.tmp.npz');np.savez_compressed(temporary,coefficients=point,model_fingerprint=model.fingerprint);temporary.replace(cp)
  if residual<1e-10:break
  old_ci=[o.cache['ci'].copy() for o in objects]
  eigenvalues,eigenvectors=np.linalg.eigh(h);projected=eigenvectors.T@g
  step=-eigenvectors@(projected/eigenvalues)
  history[-1]['unrestricted_newton_norm']=float(np.linalg.norm(step));history[-1]['minimum_absolute_curvature']=float(np.min(abs(eigenvalues)))
  radius=.3
  if np.linalg.norm(step)>radius:
   # Minimize the linearized residual in a fixed trust ball. Soft modes
   # no longer shrink every stiff component through uniform clipping.
   square=eigenvalues**2;low=0.;high=max(1.,float(np.max(square)))
   def trial_step(mu):return -eigenvectors@(eigenvalues*projected/(square+mu))
   while np.linalg.norm(trial_step(high))>radius:high*=2
   for unused in range(60):
    middle=.5*(low+high)
    if np.linalg.norm(trial_step(middle))>radius:low=middle
    else:high=middle
   step=trial_step(high)
  history[-1]['step_type']='spectral_residual_trust_newton';accepted=False
  for fraction in [1.,.5,.25,.125,.0625,.03125]:
   trial=frame.retract(point,fraction*step,rawmap,v);_,gg,*_=frame.evaluate(trial)
   guard=all(o.fidelity(c)[0]**2>.5 and o.cache['coefficient_margin']>=.05 and abs(o.signed_overlap(old,ci,c,o.cache['ci']))>=.9 for o,old,ci,c in zip(objects,point,old_ci,trial))
   if guard and np.linalg.norm(gg)<np.linalg.norm(g):point=trial;accepted=True;break
  if not accepted:raise RuntimeError('Guarded anchor continuation blocked; retained checkpoint')
  assert np.max(abs(point[0][:,frame.active].T@frame.s@point[1][:,frame.active]-target))<1e-10
 else:raise RuntimeError('Extended continuation budget exhausted; retained checkpoint')
 newdir=out/'portable_model_polished'
 if not (newdir/'model.json').exists():
  state=objects[0].context.state;chart,_=model.eri.chart(state);mapping=metric_power(state.overlap,-.5)@chart.u@metric_power(model.anchor.overlap,.5);initial=[np.linalg.solve(mapping,c) for c in point]
  write_multicsf_model(newdir,template=model.template,responses=model.responses,eri=model.arrays['eri'],initial_coefficients=initial,reference_coefficients=model.arrays['reference_coefficients'],reference_transitions=model.arrays['reference_transitions'],transition_weights=model.arrays['transition_weights'],grounds=model.grounds,provenance=dict(original_model_fingerprint=model.fingerprint,solver='same fixed full-active Gram manifold; extended guarded Newton continuation only',source_sha256=sha(__file__)))
 polished=MultiCSFModel(newdir/'model.json');assert np.max(abs(polished.target-model.target))<1e-10
 real=original.MultiCSFModel
 def selected(path):return polished if Path(path)==out/'portable_model/model.json' else real(path)
 original.MultiCSFModel=selected
 write(out/'continuation_certificate.json',dict(original_fingerprint=model.fingerprint,polished_fingerprint=polished.fingerprint,relative_frame_max_change=float(np.max(abs(polished.target-model.target))),residual=residual,source_sha256=sha(__file__),energy_model_changed=False))
 return original.run(name,sid,scale)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('name');a.add_argument('sid');a.add_argument('scale',type=float);args=a.parse_args();lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 with threadpool_limits():run(args.name,args.sid,args.scale)
