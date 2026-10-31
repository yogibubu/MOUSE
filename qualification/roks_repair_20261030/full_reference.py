"""Full common-orbital BH&HLYP ROKS references, checkpointed direct optimization.

The main optimizer needs only UKS energies and Focks. A final analytic-response
Newton correction enforces stationarity in the full orbital space. Point-group
constrained minimization is an initializer, not the final convergence test.
"""
from remote_bootstrap import HOST,BASE
from pathlib import Path
import sys,json,argparse,time,traceback,numpy as np
from scipy.linalg import expm,expm_frechet
from scipy.optimize import minimize
from scipy.sparse.linalg import gmres,LinearOperator
from pyscf import symm,lib
OLD=Path('/home/enzo/MOUSE-deltascf-validation-20261029/hybrid_spectrum_20261029');sys.path.insert(0,str(OLD))
import full_reference_engine as engine
from frozen_relaxation import write,digest,key,EV
lib.num_threads(8)
INPUT=BASE/'results/full_reference_inputs';OUT=BASE/'results/full_reference';OUT.mkdir(exist_ok=True)
PLAN=json.loads((INPUT/'frozen_relaxation/plan.json').read_text())['cases']
engine.ROOT=INPUT
engine.PROTOCOL=dict(engine.PROTOCOL,schema='full-bhandhlyp-roks-reference-v1',xc='BHANDHLYP',
 cases=[c['name'] for c in PLAN],threads=8,orbital_tolerance=1e-7,
 solver='Cs/C2v L-BFGS initializer with exact chart gradient; full-space Newton certificate',
 maximum_minimizer_iterations=160,maximum_final_newton_steps=16)

def save_point(path,obj,v):
 tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,request_hash=obj.request,coefficients=v['c'],
  energy_hartree=v['energy'],gradient=v['g']);tmp.replace(path)

def run(case,state):
 out=OUT/case/state;out.mkdir(parents=True,exist_ok=True);final=out/'result.json'
 if final.exists():
  old=json.loads(final.read_text())
  if old.get('qualified'):print('REUSE FULL',case,state,flush=True);return
 obj=engine.Objective(case,state);cp=out/'latest.npz';qualified_cp=out/'last_qualified.npz'
 start=time.perf_counter();c=obj.C0.copy()
 if cp.exists():
  with np.load(cp) as f:assert str(f['request_hash'])==obj.request;c=f['coefficients'].copy()
 if obj.overlap(obj.C0,c)<.7 and qualified_cp.exists():
  with np.load(qualified_cp) as f:assert str(f['request_hash'])==obj.request;c=f['coefficients'].copy()
 obj.mol.symmetry=True;obj.mol.build(False,False)
 assert obj.mol.groupname in ['Cs','C2v']
 labels=symm.label_orb_symm(obj.mol,obj.mol.irrep_name,obj.mol.symm_orb,obj.C0,s=obj.S,check=True)
 mix=obj.C0.T@obj.S@c;off=labels[:,None]!=labels[None,:]
 assert np.max(abs(mix[off]))<1e-4
 for label in set(labels):
  ids=np.flatnonzero(labels==label);u,_,vt=np.linalg.svd(mix[np.ix_(ids,ids)]);mix[np.ix_(ids,ids)]=u@vt
 mix[off]=0;c=obj.C0@mix
 opens=np.flatnonzero(obj.occ[0]!=obj.occ[1])
 active=np.flatnonzero((labels[obj.i]==labels[obj.j]) & ~np.isin(obj.i,opens) & ~np.isin(obj.j,opens))
 i,j=obj.i[active],obj.j[active]
 initial=obj.evaluate(c);diag=np.zeros(obj.size)
 for factor,occ,fs in zip((2.,-1.),(obj.occ,obj.trip),initial['f']):
  for o,f in zip(occ,fs):
   d=np.diag(f);diag+=2*factor*(d[obj.i]-d[obj.j])*(o[obj.j]-o[obj.i])
 scale=np.sqrt(np.clip(abs(diag[active]),.05,200.));cache={}
 def fun(y):
  if 'y' in cache and np.array_equal(y,cache['y']):return cache['energy'],cache['gradient']
  k=np.zeros((obj.n,obj.n));k[i,j]=y/scale;k[j,i]=-y/scale;u=expm(k)
  v=obj.evaluate(c@u);g=np.zeros((obj.n,obj.n));g[i,j]=v['g'][active]/2;g[j,i]=-v['g'][active]/2
  chart=expm_frechet(k.T,u@g,compute_expm=False);grad=(chart[i,j]-chart[j,i])/scale
  cache.update(y=y.copy(),energy=v['energy'],gradient=grad,value=v);return v['energy'],grad
 audit=out/'gradient_audit.json'
 if not audit.exists():
  rng=np.random.default_rng(20261030);test=rng.normal(size=len(i));test*=.02/np.linalg.norm(test)
  direction=rng.normal(size=len(i));direction/=np.linalg.norm(direction);h=1e-4
  _,g=fun(test);ep,_=fun(test+h*direction);em,_=fun(test-h*direction)
  error=abs((ep-em)/(2*h)-g@direction)
  write(audit,dict(host=HOST,request_hash=obj.request,chart_step=h,energy_direction_error=float(error),passed=bool(error<2e-7)))
  assert error<2e-7
 else:
  a=json.loads(audit.read_text());assert a['request_hash']==obj.request and str(a['passed']).lower()=='true'
 history_path=out/'history.json';history=json.loads(history_path.read_text()) if history_path.exists() else []
 class InitializerComplete(Exception):pass
 def callback(y):
  fun(y);v=cache['value'];score=obj.overlap(obj.C0,v['c'])
  row=dict(stage='variational_initializer',iteration=len(history),energy_hartree=v['energy'],
   excitation_ev=(v['energy']-obj.ground)*EV,residual=float(np.max(abs(v['g']))),csf_overlap=score,
   seconds_current_run=time.perf_counter()-start)
  history.append(row);write(history_path,history);save_point(cp,obj,v)
  if score>=.7:save_point(qualified_cp,obj,v)
  print('FULL',case,state,row,flush=True)
  if score<.7 or np.max(abs(v['g'][active]),initial=0.)<2e-6:raise InitializerComplete()
 try:
  solved=minimize(fun,np.zeros(len(i)),method='L-BFGS-B',jac=True,callback=callback,
   options=dict(gtol=2e-8,ftol=2e-16,maxiter=160,maxls=20,maxcor=12))
  fun(solved.x)
 except InitializerComplete:pass
 v=cache['value']
 if obj.overlap(obj.C0,v['c'])<.7:
  with np.load(qualified_cp) as f:v=obj.evaluate(f['coefficients'])
 for iteration in range(16):
  g=v['g'];norm=float(np.linalg.norm(g));error=float(np.max(abs(g)))
  if error<=1e-7:break
  hv,diagonal,calls=obj.response(v)
  a=LinearOperator((obj.size,obj.size),matvec=hv,dtype=float)
  pre=LinearOperator(a.shape,matvec=lambda x:x/diagonal,dtype=float)
  step,info=gmres(a,-g,M=pre,rtol=.08,atol=0,restart=24,maxiter=2)
  step*=min(1.,.2/max(np.linalg.norm(step),1e-30));accepted=False
  for fraction in [1.,.5,.25,.125,.0625]:
   trial=obj.rotate(v['c'],fraction*step)
   if obj.overlap(obj.C0,trial)<.7:continue
   new=obj.evaluate(trial)
   if np.linalg.norm(new['g'])<norm*(1-1e-4*fraction):v=new;accepted=True;break
  if not accepted:raise RuntimeError('full-space Newton certification stalled; checkpoint retained')
  save_point(cp,obj,v);save_point(qualified_cp,obj,v)
  row=dict(stage='full_space_newton',iteration=iteration,energy_hartree=v['energy'],
   excitation_ev=(v['energy']-obj.ground)*EV,residual=float(np.max(abs(v['g']))),
   csf_overlap=obj.overlap(obj.C0,v['c']),response_products=calls[0],gmres_info=int(info))
  history.append(row);write(history_path,history);print('CERTIFY',case,state,row,flush=True)
 error=float(np.max(abs(v['g'])));score=obj.overlap(obj.C0,v['c'])
 assert error<=1e-7 and score>=.7
 save_point(cp,obj,v)
 result=dict(host=HOST,molecule=case,state=state,request_hash=obj.request,qualified=True,
  functional='BHANDHLYP',basis='aug-cc-pVDZ Cartesian',protocol=engine.PROTOCOL,
  nao=len(obj.S),nmo=obj.n,group=obj.mol.groupname,active_full_rotations=obj.size,initializer_rotations=len(i),
  energy_hartree=v['energy'],ground_hartree=obj.ground,mixed_energy_hartree=v['mixed_energy'],
  triplet_energy_hartree=v['triplet_energy'],excitation_ev=(v['energy']-obj.ground)*EV,
  tbe_ev=obj.row['tbe_ev'],error_ev=(v['energy']-obj.ground)*EV-obj.row['tbe_ev'],
  full_orbital_residual=error,seed_csf_overlap=score,spin_squared_csf=0.,
  hessian_index_certified=False,seconds_current_run=time.perf_counter()-start,
  initializer_constraints='point group and both open orbitals frozen; all active rotations released for Newton',
  ground_reference_reused=True,script_sha256=digest(__file__))
 write(final,result);print('FULL RESULT',case,state,result,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--molecule');p.add_argument('--group',type=int,default=0);p.add_argument('--groups',type=int,default=1);args=p.parse_args()
 write(OUT/f'protocol_group_{args.group}.json',engine.PROTOCOL)
 for k,case in enumerate(PLAN):
  if k%args.groups!=args.group or (args.molecule and case['name']!=args.molecule):continue
  for target in case['targets']:
   try:run(case['name'],target['id'])
   except Exception as exc:
    write(OUT/case['name']/target['id']/'failure.json',dict(host=HOST,error=str(exc),traceback=traceback.format_exc()))
    print(traceback.format_exc(),flush=True)
