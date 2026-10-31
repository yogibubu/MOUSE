"""Energy-only feasibility: retain the assigned parents' relative active angles.

Each CSF is spin pure. The energy is precisely the independent 50/50 energy;
only their relative open-orbital squared overlaps are held at the ANCHOR
values. Thus completed vertical energies are unchanged, without an offset.
This prototype exposes no nuclear derivative API.
"""
import numpy as np
from scipy.linalg import expm, null_space
from types import SimpleNamespace
from matrix_gfn2.roks_orbital_response import orbital_residual_jacobian

class RelativePairObjective:
 def __init__(self,contexts,occ,target):
  self.contexts=contexts;self.occ=np.asarray(occ);self.triplet=np.array([np.maximum(*occ),np.minimum(*occ)])
  self.s=contexts[0].state.overlap;self.n=len(self.s);l,r=np.triu_indices(self.n,1);labels=np.vstack([self.occ,self.triplet]);keep=np.any(labels[:,l]!=labels[:,r],axis=0);self.l,self.r=l[keep],r[keep]
  self.m=len(self.l);self.dimension=2*self.m;self.generators=np.zeros((self.m,self.n,self.n))
  for k,(i,j) in enumerate(zip(self.l,self.r)):self.generators[k,i,j]=1.;self.generators[k,j,i]=-1.
  self.opens=np.flatnonzero(self.occ[0]!=self.occ[1]);self.target=np.asarray(target);self.cache={}
 def retract(self,point,step):
  return np.asarray([c@expm(np.einsum('a,aij->ij',v,self.generators)) for c,v in zip(point,np.reshape(step,(2,-1)))])
 def evaluate(self,point,mu,curvature=True):
  if 'point' not in self.cache or not np.array_equal(self.cache['point'],point):
   e=0.;gg=[];focks=[]
   for c,ctx in zip(point,self.contexts):
    g=np.zeros(self.m);ff=[]
    for weight,o in [(2.,self.occ),(-1.,self.triplet)]:
     f,energy=ctx.fock_and_energy(np.array([(c*x)@c.T for x in o]));e+=.5*weight*energy;ff.append(f)
     q=sum((c.T@v@c)*x[None,:] for v,x in zip(f,o));g+=weight*2*(q[self.l,self.r]-q[self.r,self.l])
    gg.append(.5*g);focks.append(ff)
   self.cache=dict(point=point.copy(),energy=e,g=np.concatenate(gg),focks=focks)
  if curvature and 'h' not in self.cache:
   h=np.zeros((self.dimension,self.dimension))
   for i,(c,ctx,ff) in enumerate(zip(point,self.contexts,self.cache['focks'])):
    state=SimpleNamespace(**vars(ctx.state));state.coefficients=c;state.alpha_occupations,state.beta_occupations=self.occ;kernel=ctx.density_response_kernel(state)
    block=sum(.5*w*orbital_residual_jacobian(c,o,f,kernel,ctx.polarization,self.l,self.r) for w,o,f in [(2.,self.occ,ff[0]),(-1.,self.triplet,ff[1])])
    h[i*self.m:(i+1)*self.m,i*self.m:(i+1)*self.m]=.5*(block+block.T)
   self.cache['h']=h
  h=self.cache['h'].copy() if curvature else None;cvals=[];jac=[]
  ca,cb=point
  for k,multiplier,target in zip(self.opens,mu,self.target):
   a,b=ca[:,k],cb[:,k];overlap=float(a@self.s@b);col=self.generators[:,:,k];za,zb=ca.T@self.s@b,cb.T@self.s@a
   dr=np.concatenate([col@za,col@zb]);cvals.append(overlap**2-target);jac.append(2*overlap*dr)
   if curvature:
    second=np.zeros_like(h)
    for i,z in enumerate([za,zb]):
     zg=np.einsum('i,aij->aj',z,self.generators);ss=zg@col.T;second[i*self.m:(i+1)*self.m,i*self.m:(i+1)*self.m]=.5*(ss+ss.T)
    cross=(ca@col.T).T@self.s@(cb@col.T);second[:self.m,self.m:]=cross;second[self.m:,:self.m]=cross.T
    h+=multiplier*(2*np.outer(dr,dr)+2*overlap*second)
  jac=np.asarray(jac);res=np.concatenate([self.cache['g']+jac.T@mu,cvals]);kkt=None if h is None else np.block([[h,jac.T],[jac,np.zeros((2,2))]])
  return self.cache['energy'],res,kkt,jac

def stationary_continuation(objective,point,checkpoint):
 final=objective.target.copy();initial=np.array([(point[0][:,k]@objective.s@point[1][:,k])**2 for k in objective.opens]);mu=np.zeros(2);fraction=0.;advance=.1;history=[]
 while fraction<1.-1e-12:
  trial_fraction=min(1.,fraction+advance);objective.target=initial+trial_fraction*(final-initial);start=point.copy();oldmu=mu.copy();ok=False
  for iteration in range(81):
   energy,res,kkt,jac=objective.evaluate(point,mu);error=float(np.max(abs(res)))
   checkpoint(point,mu,trial_fraction,iteration,error)
   if error<1e-9:ok=True;break
   step=np.linalg.lstsq(kkt,-res,rcond=1e-12)[0];norm=np.linalg.norm(step[:objective.dimension])
   if norm>.1:step*=.1/norm
   for scale in [1.,.5,.25,.125,.0625,.03125,.015625,.0078125]:
    trial=objective.retract(point,scale*step[:objective.dimension]);mt=mu+scale*step[objective.dimension:];_,rr,_,_=objective.evaluate(trial,mt,curvature=False)
    if np.linalg.norm(rr)<np.linalg.norm(res):point,mu=trial,mt;break
   else:break
  if not ok:
   point,mu=start,oldmu;advance*=.5
   print('ADAPT',trial_fraction,error,advance,flush=True)
   if advance<1e-5:raise RuntimeError('relative-pair continuation requires branch handling')
   continue
  fraction=trial_fraction;advance=min(.2,advance*1.3)
  z=null_space(jac);values=np.linalg.eigvalsh(z.T@kkt[:objective.dimension,:objective.dimension]@z)
  row=dict(fraction=fraction,residual=error,iterations=iteration,index=int(np.sum(values<-1e-7)),energy=energy)
  history.append(row);print('RELATIVE',row,flush=True)
 objective.target=final
 return point,mu,history
