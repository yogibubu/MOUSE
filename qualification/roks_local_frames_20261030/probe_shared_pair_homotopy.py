"""Variational fixed-50/50 ROKS with a shared active pair and independent spectators.

No penalty parameter or energy shift. Closed/virtual spectators relax separately
for the two parent Hamiltonians. The two open spatial orbitals are identical.
The coupled analytic orbital Hessian includes the product-chart BCH terms.
This feasibility probe deliberately exposes NO nuclear derivative API.
"""
import os,socket,sys,json,time,hashlib
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
from scipy.linalg import expm
import numpy as np
BASE=Path('/home/enzo/MOUSE-roks-local-20261030');OLD=Path('/home/enzo/MOUSE-roks-independent-20261030');OUT=Path('/home/enzo/MOUSE-roks-shared-pair-20261030')
assert socket.getfqdn()=='oracle.sns.it';os.environ['MOUSE_VALIDATION_ROOT']=str(BASE)
sys.path.insert(0,str(OLD));import remote_bootstrap
for path in (BASE/'source/packages').glob('*/src'):sys.path.insert(0,str(path))
from threadpoolctl import threadpool_limits
from matrix_gfn2 import GFN2Calculator,load_response_reference,FrozenNonlocalResponseReference
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.roks_orbital_response import orbital_residual_jacobian
from matrix_gfn2.deltascf_derivatives import load_derivative_point
from matrix_gfn2.roks_tracking import active_orbital_character
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR

def load(path,request=None):
 with np.load(path) as d:xyz=d['coordinates_angstrom'].copy();req=str(d['request_hash']) if request is None else request
 return load_derivative_point(path,request_hash=req,coordinates_angstrom=xyz,require_gradient=False)

def root_metric(s,power):
 w,v=np.linalg.eigh(s);return (v*w**power)@v.T

class SharedPairObjective:
 def __init__(self,contexts,occupations):
  self.contexts=contexts;self.occ=np.asarray(occupations);self.triplet=np.array([np.maximum(*self.occ),np.minimum(*self.occ)])
  self.n=len(self.occ[0]);self.s=contexts[0].state.overlap;self.left,self.right=np.triu_indices(self.n,1)
  self.opens=np.flatnonzero(self.occ[0]!=self.occ[1]);assert len(self.opens)==2
  shared=np.isin(self.left,self.opens)|np.isin(self.right,self.opens)
  private=(~shared)&(self.occ.sum(axis=0)[self.left]!=self.occ.sum(axis=0)[self.right])
  self.shared=np.flatnonzero(shared);self.private=np.flatnonzero(private);self.ns=len(self.shared);self.np=len(self.private);self.dimension=self.ns+2*self.np
  self.generators=np.zeros((len(self.left),self.n,self.n))
  for a,(i,j) in enumerate(zip(self.left,self.right)):self.generators[a,i,j]=1.;self.generators[a,j,i]=-1.
  self.cache={}
 def seed(self,components):
  root=root_metric(self.s,.5);inverse=root_metric(self.s,-.5);q=[root@c for c in components];opens=self.opens;others=np.array([i for i in range(self.n) if i not in opens])
  for i in range(2):
   if q[0][:,opens[i]]@q[1][:,opens[i]]<0:q[1][:,opens[i]]*=-1
  mean=q[0][:,opens]+q[1][:,opens];u,_,vt=np.linalg.svd(mean,full_matrices=False);common=u@vt
  result=[]
  for qq in q:
   rest=qq[:,others]-common@(common.T@qq[:,others]);u,_,vt=np.linalg.svd(rest,full_matrices=False)
   qq[:,others]=u@vt;qq[:,opens]=common;result.append(inverse@qq)
  return np.asarray(result)
 def maps(self,point):
  a,b=point;v=a.T@self.s@b
  shared=self.generators[self.shared];hs=[shared,v.T@shared@v]
  maps=[np.zeros((len(self.left),self.dimension)) for _ in range(2)]
  for i in range(2):
   maps[i][:,:self.ns]=hs[i][:,self.left,self.right].T
   maps[i][self.private,self.ns+i*self.np+np.arange(self.np)]=1.
  return maps,hs,v
 def retract(self,point,step):
  a,b=point;_,_,v=self.maps(point);g=np.einsum('a,aij->ij',step[:self.ns],self.generators[self.shared]);common=expm(g)
  pa=expm(np.einsum('a,aij->ij',step[self.ns:self.ns+self.np],self.generators[self.private]));pb=expm(np.einsum('a,aij->ij',step[self.ns+self.np:],self.generators[self.private]))
  return np.asarray([a@common@pa,a@common@v@pb])
 def evaluate(self,point,curvature=True):
  if 'point' not in self.cache or not np.array_equal(point,self.cache['point']):
   values=[]
   for c,ctx in zip(point,self.contexts):
    focks=[];energy=0.;gradient=np.zeros(len(self.left))
    for weight,o in [(2.,self.occ),(-1.,self.triplet)]:
     dm=np.array([(c*x)@c.T for x in o]);f,e=ctx.fock_and_energy(dm);energy+=weight*e;focks.append(f)
     q=sum((c.T@ff@c)*x[None,:] for ff,x in zip(f,o));gradient+=weight*2*(q[self.left,self.right]-q[self.right,self.left])
    values.append(dict(energy=energy,g=gradient,focks=focks))
   maps,hs,v=self.maps(point);g=.5*sum(t.T@row['g'] for t,row in zip(maps,values));self.cache.clear();self.cache.update(point=point.copy(),values=values,maps=maps,hs=hs,v=v,g=g,energy=.5*sum(row['energy'] for row in values))
  if curvature and 'h' not in self.cache:
   total=np.zeros((self.dimension,self.dimension))
   for model,(c,ctx,row,t,hs) in enumerate(zip(point,self.contexts,self.cache['values'],self.cache['maps'],self.cache['hs'])):
    state=SimpleNamespace(**vars(ctx.state));state.coefficients=c;state.alpha_occupations,state.beta_occupations=self.occ
    kernel=ctx.density_response_kernel(state);h=np.zeros((len(self.left),len(self.left)))
    for weight,o,f in [(2.,self.occ,row['focks'][0]),(-1.,self.triplet,row['focks'][1])]:
     h+=weight*orbital_residual_jacobian(c,o,f,kernel,ctx.polarization,self.left,self.right)
    h=.5*(h+h.T);row['h']=h;total+=.5*t.T@h@t
    # E(exp(Hs) exp(Hp)): mixed second chart derivative includes 1/2[Hs,Hp].
    gg=np.zeros((self.n,self.n));gg[self.left,self.right]=row['g'];gg[self.right,self.left]=-row['g']
    comm=gg@hs-hs@gg
    correction=.25*comm[:,self.left[self.private],self.right[self.private]]
    columns=self.ns+model*self.np+np.arange(self.np)
    total[np.ix_(np.arange(self.ns),columns)]+=correction
    total[np.ix_(columns,np.arange(self.ns))]+=correction.T
   self.cache['h']=.5*(total+total.T)
  return self.cache['energy'],self.cache['g'],self.cache.get('h')

class PenaltyObjective:
 def __init__(self,base,xi):
  self.base=base;self.xi=xi
  labels=np.vstack([base.occ,base.triplet]);self.active=np.flatnonzero(np.any(labels[:,base.left]!=labels[:,base.right],axis=0));self.generators=base.generators[self.active];self.dimension=len(self.active)
 def retract(self,point,step):
  return np.array([c@expm(np.einsum('a,aij->ij',d,self.generators)) for c,d in zip(point,np.reshape(step,(2,-1)))])
 def evaluate(self,point,curvature=True):
  self.base.evaluate(point,curvature=curvature);rows=self.base.cache['values'];m=self.dimension
  e=.5*sum(r['energy'] for r in rows);g=.5*np.concatenate([r['g'][self.active] for r in rows]);h=None
  if curvature:
   h=np.zeros((2*m,2*m))
   for i,row in enumerate(rows):h[i*m:(i+1)*m,i*m:(i+1)*m]=.5*row['h'][np.ix_(self.active,self.active)]
  ca,cb=point;s=self.base.s;square=root_metric(s,.5)
  for k in self.base.opens:
   a,b=ca[:,k],cb[:,k];r=float(a@s@b);columns=self.generators[:,:,k]
   qa,qb=square@a,square@b;difference=np.outer(qa,qa)-np.outer(qb,qb);e+=.5*self.xi*np.sum(difference*difference)
   za,zb=ca.T@s@b,cb.T@s@a;ga,gb=columns@za,columns@zb;dr=np.concatenate([ga,gb]);g-=2*self.xi*r*dr
   if curvature:
    second=np.zeros_like(h)
    for i,z in enumerate([za,zb]):
     zg=np.einsum('i,aij->aj',z,self.generators);ss=zg@columns.T;second[i*m:(i+1)*m,i*m:(i+1)*m]=.5*(ss+ss.T)
    da=(ca@columns.T).T;db=(cb@columns.T).T;cross=da@s@db.T;second[:m,m:]=cross;second[m:,:m]=cross.T
    h-=2*self.xi*(np.outer(dr,dr)+r*second)
  return float(e),g,None if h is None else .5*(h+h.T)

def run(name):
 root=OUT/'results'/name/'coupling_continuation';root.mkdir(parents=True,exist_ok=True)
 if (root/'result.json').exists():return
 parents=[];contexts=[];anchors=[];ground=[]
 if name=='formamide':
  data=json.loads((BASE/'results'/name/'prepared.json').read_text());manifest=json.loads((BASE/'results'/name/'optimization/roks_50_50_accepted/accepted.json').read_text());accepted=BASE/'results'/name/'optimization/roks_50_50_accepted'
  for i,row in enumerate(manifest['reference']):parents.append(load(accepted/row['path']))
  for i in range(2):anchors.append(load(BASE/'results'/name/f'anchor_{i}.npz'));ground.append(load(accepted/manifest['ground_references'][i+1]['path']).energy_hartree)
  refs=[load_response_reference(p) for p in data['response_paths']]
 else:
  refs=[]
  for profile in ['full_response','qualified_spectral_response']:
   report=json.loads((OLD/'results'/profile/name/'n_pi/result.json').read_text());assert report['qualified'];parent=load(Path(report['checkpoint']),report['request_hash']);parents.append(parent);anchors.append(parent);ground.append(parent.ground_energy_hartree)
   state=parent.reference_state;atoms=[{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in state.atomic_numbers];calc=GFN2Calculator(atoms,temperature_kelvin=0.)
   native=prepare_gfn2_deltascf(calc,state.coordinates_bohr/ANGSTROM_TO_BOHR,orbital_reference=state)
   path=OLD/'results'/profile/name/'model.npz'
   with np.load(path) as d:model={k:d[k].copy() for k in ['operators','charge','spin','q0','linear']}
   refs.append(FrozenNonlocalResponseReference.from_vertical_model(state,native.kernel.partition,model,hashlib.sha256(path.read_bytes()).hexdigest()))
  data=dict(atoms=atoms)
 for parent,ref in zip(parents,refs):
  calc=GFN2Calculator(data['atoms'],temperature_kelvin=0.);calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('SCC invoked'))
  contexts.append(prepare_gfn2_deltascf(calc,parent.reference_state.coordinates_bohr/ANGSTROM_TO_BOHR,orbital_reference=parent.reference_state,spin_reference=ref))
 objective=SharedPairObjective(contexts,parents[0].occupations);point=np.asarray([r.common_coefficients for r in parents]);start=time.monotonic();history=[];phases=[]
 for xi in np.logspace(-4,4,17):
  path=root/f'xi_{xi:.8g}.npz'
  if path.exists():
   with np.load(path) as d:point=d['coefficients'].copy()
   continue
  penalty=PenaltyObjective(objective,float(xi));tol=max(1e-9,float(xi)*1e-12)
  for step_index in range(101):
   e,g,h=penalty.evaluate(point);error=float(np.max(abs(g)))
   np.savez_compressed(root/'continuation_latest.npz',coefficients=point,xi=xi,energy=e,residual=error,iteration=step_index)
   if error<tol:break
   delta=np.linalg.lstsq(h,-g,rcond=1e-12)[0];norm=np.linalg.norm(delta)
   if norm>.15:delta*=.15/norm
   for fraction in [1.,.5,.25,.125,.0625,.03125,.015625]:
    trial=penalty.retract(point,fraction*delta);_,gg,_=penalty.evaluate(trial,curvature=False)
    if np.linalg.norm(gg)<np.linalg.norm(g):point=trial;break
   else:raise RuntimeError('coupling continuation stalled: '+str(dict(xi=xi,residual=error)))
  else:raise RuntimeError('coupling continuation iteration limit')
  if abs(np.log10(xi))<1e-10:
   direction=np.sin(np.arange(len(g))+.2);direction/=np.linalg.norm(direction);eps=1e-4
   _,gm,_=penalty.evaluate(penalty.retract(point,-eps*direction),curvature=False);_,gp,_=penalty.evaluate(penalty.retract(point,eps*direction),curvature=False)
   assert np.max(abs((gp-gm)/(2*eps)-h@direction))<1e-6,'coupling Hessian check failed'
  overlap=point[0][:,objective.opens].T@objective.s@point[1][:,objective.opens]
  row=dict(xi=float(xi),residual=error,index=int(np.sum(np.linalg.eigvalsh(h)<-1e-7)),open_overlaps=np.diag(overlap).tolist(),iterations=step_index)
  phases.append(row);print('COUPLING',json.dumps(row),flush=True);np.savez_compressed(path,coefficients=point,xi=xi,residual=error)
  (root/'coupling_trace.json').write_text(json.dumps(phases,indent=2)+'\n')
 point=objective.seed(point)
 for iteration in range(101):
  e,g,h=objective.evaluate(point);error=float(np.max(abs(g)));values=np.linalg.eigvalsh(h);index=int(np.sum(values<-1e-7))
  history.append(dict(iteration=iteration,energy=e,residual=error,index=index,minimum_absolute_curvature=float(np.min(abs(values)))))
  np.savez_compressed(root/'latest.npz',coefficients=point,occupations=objective.occ,history=json.dumps(history),energy=e,index=index)
  if error<1e-9 and np.min(abs(values))>1e-7:break
  delta=np.linalg.lstsq(h,-g,rcond=1e-10)[0];norm=np.linalg.norm(delta)
  if norm>.15:delta*=.15/norm
  for fraction in [1.,.5,.25,.125,.0625,.03125]:
   trial=objective.retract(point,fraction*delta);_,gg,_=objective.evaluate(trial,curvature=False)
   if np.linalg.norm(gg)<np.linalg.norm(g):point=trial;break
  else:raise RuntimeError('joint orbital Newton stalled: '+str(history[-1]))
 else:raise RuntimeError('joint orbital iteration limit')
 e,g,h=objective.evaluate(point);direction=np.sin(np.arange(objective.dimension)+.3);direction/=np.linalg.norm(direction);checks=[]
 for step in [2e-4,1e-4]:
  em,gm,_=objective.evaluate(objective.retract(point,-step*direction),curvature=False);ep,gp,_=objective.evaluate(objective.retract(point,step*direction),curvature=False)
  checks.append(dict(step=step,gradient_error=float(abs((ep-em)/(2*step)-g@direction)),hvp_error=float(np.max(abs((gp-gm)/(2*step)-h@direction)))))
 objective.evaluate(point);characters=[]
 for i,(parent,anchor,ref,c,row) in enumerate(zip(parents,anchors,refs,point,objective.cache['values'])):
  descriptor=replace(parent,common_coefficients=c,spin_reference_fingerprint=ref.fingerprint,spin_frame_clusters=getattr(ref,'frame_clusters',None),converged=False,orbital_saddle_index=None,target_orbital_saddle_index=None)
  anchor=replace(anchor,spin_reference_fingerprint=ref.fingerprint,spin_frame_clusters=getattr(ref,'frame_clusters',None))
  characters.append(active_orbital_character(anchor,descriptor,ref.anchor_coordinates_bohr))
 out=dict(molecule=name,excitation_ev=(e-.5*sum(ground))*27.211386245988,independent_excitation_ev=(.5*sum(r.energy_hartree for r in parents)-.5*sum(ground))*27.211386245988,joint_residual=float(np.max(abs(g))),joint_index=index,characters=characters,checks=checks,component_unconstrained_residuals=[float(np.max(abs(r['g']))) for r in objective.cache['values']],shared_pair_error=float(np.max(abs(point[0][:,objective.opens]-point[1][:,objective.opens]))),seconds=time.monotonic()-start,iterations=iteration,nuclear_derivatives=False,zero_offset=True,weights=[.5,.5])
 out['qualified']=all(c['transition']>=.7 for c in characters) and out['shared_pair_error']<1e-10 and all(r['gradient_error']<1e-7 and r['hvp_error']<1e-6 for r in checks)
 (root/'result.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)

if __name__=='__main__':
 with threadpool_limits(limits=2):run(sys.argv[1])
