"""Oracle-only orbital optimization of the predeclared minimal CSF candidate.
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
 out=OUT/'optimized'/name/sid/(str(index)+'_gmf_polish_v1');out.mkdir(parents=True,exist_ok=True)
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
 if not latest.exists():
  import shutil
  seed=OUT/'retained_gmf_seed'/f'{sid}_{index}.npz'
  shutil.copy2(seed,latest)
  write(out/'seed_provenance.json',dict(source=str(seed),source_sha256=hashlib.sha256(seed.read_bytes()).hexdigest(),reason='resume resolved-index GMF checkpoint with full analytic Newton step; no reset'))
 if latest.exists():
  with np.load(latest) as a:assert str(a['model_fingerprint'])==ref.fingerprint;c=a['coefficients'].copy()
 history=[];evaluations=0;status='unresolved';reason='iteration budget'
 for iteration in range(60):
  e,g,h=obj.evaluate(c,True);evaluations+=1;ci=obj.cache['ci'].copy();fidelity,raw=obj.fidelity(c);error=float(np.max(abs(g)));values=np.linalg.eigvalsh(h)
  history.append(dict(iteration=iteration,energy_ev=(e-e0)*EV,residual=error,index=int(np.count_nonzero(values< -1e-7)),minimum_absolute_curvature=float(np.min(abs(values))),fidelity=fidelity,raw_transition=raw,ci=ci.tolist()))
  np.savez_compressed(latest,coefficients=c,ci=ci,model_fingerprint=ref.fingerprint)
  write(out/'history.json',history)
  if error<=1e-9:
   status='stationary_candidate' if np.min(abs(values))>1e-7 else 'index_unresolved';break
  if time.monotonic()-start>210:reason='time budget; checkpoint retained';break
  step=np.linalg.lstsq(h,-g,rcond=1e-10)[0];norm=np.linalg.norm(step)
  if norm>.08:step*=.08/norm
  accepted=False;blocked=[]
  for fraction in [1.,.5,.25,.125,.0625,.03125]:
   k=np.zeros_like(c);k[obj.left,obj.right]=fraction*step;k[obj.right,obj.left]=-fraction*step;trial=c@expm(k)
   ee,gg,_=obj.evaluate(trial,False);evaluations+=1;trial_ci=obj.cache['ci'].copy();ff,rr=obj.fidelity(trial)
   overlap=abs(obj.signed_overlap(c,ci,trial,trial_ci));sectors=min(np.min(np.linalg.norm(c0[:,:no].T@s.overlap@trial[:,:rank],axis=0)),np.min(np.linalg.norm(c0[:,no:].T@s.overlap@trial[:,no:no+rank],axis=0)))
   guard=ff>np.sqrt(.5) and overlap>=.90 and sectors>=.7 and obj.cache['coefficient_margin']>=.05
   if guard and np.linalg.norm(gg)<np.linalg.norm(g):c=trial;accepted=True;break
   blocked.append(dict(fraction=fraction,fidelity=ff,overlap=overlap,sectors=float(sectors),gradient_norm=float(np.linalg.norm(gg)),ci_margin=obj.cache['coefficient_margin']))
  if not accepted:reason='Newton step blocked or no residual reduction';write(out/'blocked_steps.json',blocked);break
 e,g,h=obj.evaluate(c,True);values=np.linalg.eigvalsh(h);fidelity,raw=obj.fidelity(c)
 result=dict(host=socket.getfqdn(),molecule=name,state=sid,profile=index,status=status,reason=None if status=='stationary_candidate' else reason,rank=rank,rank_selection='minimum rank retaining 95 percent of original fixed projected transition; no energy used',retained_weight=float(np.sum(native_sv[:rank]**2)),excitation_ev=float((e-e0)*EV),residual=float(np.max(abs(g))),orbital_index=int(np.count_nonzero(values< -1e-7)),minimum_absolute_curvature=float(np.min(abs(values))),fidelity=fidelity,raw_transition=raw,projection_norm=pn*target['reference_transition_norm'],ci=obj.cache['ci'].tolist(),seconds=time.monotonic()-start,evaluations=evaluations,model_fingerprint=ref.fingerprint,offdiagonal='unscaled bare projected 2(ia|jb)-(ij|ab); different multistate functional, not a changed TDA kernel',diagonal_lambda_exchange=.9,energy_offset_ev=0.,weights=[.5,.5],nuclear_derivatives_available=False,accuracy_labels_already_exposed=True)
 write(out/'result.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
 with threadpool_limits(limits=2):main(sys.argv[1],sys.argv[2],int(sys.argv[3]))
