"""Constrained, spin-pure ROKS feasibility with exact many-body reference orthogonality.

One occupied alpha principal vector lies in the reference virtual space.
This enforces zero many-body CSF/ground overlap without freezing an orbital. A KKT stationary solve uses analytic
orbital energy/constraint Hessians. Frozen and current ground projectors are
reported separately. This energy-only probe has NO nuclear derivative API.
"""
import importlib.util,json,time
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
from scipy.linalg import expm,null_space
BASE=Path('/home/enzo/MOUSE-roks-direct-20261030')
spec=importlib.util.spec_from_file_location('p',BASE/'source/qualification/probe.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
from matrix_gfn2.deltascf_derivatives import load_derivative_point
from matrix_gfn2.restricted_ground import evaluate_restricted_ground
from matrix_gfn2.roks_orbital_response import orbital_residual_jacobian
from matrix_gfn2.nonlocal_response import TransportedOperators
from matrix_gfn2.roks_tracking import active_orbital_character
np=p.np

def metric_root(s,power):
 w,v=np.linalg.eigh(s);return (v*w**power)@v.T

def solve(ctx,initial,occ,qocc,root,label):
 n=len(initial);l,r=np.triu_indices(n,1);triplet=np.array([np.maximum(*occ),np.minimum(*occ)])
 labels=np.vstack([occ,triplet]);keep=np.any(labels[:,l]!=labels[:,r],axis=0);l,r=l[keep],r[keep]
 generators=np.zeros((len(l),n,n))
 for a,(i,j) in enumerate(zip(l,r)):generators[a,i,j]=1.;generators[a,j,i]=-1.
 particle=int(np.flatnonzero(occ[0]-occ[1]==1)[0]);others=np.array([i for i in range(n) if i!=particle]);square=metric_root(ctx.state.overlap,.5)
 alpha=np.flatnonzero(occ[0]>.5);beta=np.flatnonzero(occ[1]>.5)
 inverse=metric_root(ctx.state.overlap,-.5)
 def project(c):
  q=square@c;occupied=q[:,alpha];a=qocc.T@occupied;u,sv,vt=np.linalg.svd(a)
  x=occupied@vt[-1];v=x-qocc@(qocc.T@x);norm=np.linalg.norm(v)
  if norm<1e-8:raise ValueError('reference and excited occupied subspaces coincide')
  v/=norm;cosine=float(np.clip(x@v,-1,1));sine=np.sqrt(max(0.,1-cosine*cosine))
  if sine>1e-12:
   w=(v-cosine*x)/sine
   rotation=np.eye(n)+(cosine-1)*(np.outer(x,x)+np.outer(w,w))+sine*(np.outer(w,x)-np.outer(x,w))
   q=rotation@q
  return inverse@q
 c=project(initial);mu=np.zeros(1);history=[]
 def determinant_jet(d,first,second=None):
  u,sv,vt=np.linalg.svd(d);sign=np.linalg.det(u)*np.linalg.det(vt)
  f=float(sign*np.prod(sv));dd=u.T@first@vt.T
  weights=np.array([np.prod(np.delete(sv,i)) for i in range(len(sv))])*sign
  gradient=np.einsum('i,aii->a',weights,dd)
  if second is None:return f,gradient,None
  d2=u.T@second@vt.T;h=np.einsum('i,abii->ab',weights,d2)
  for i in range(len(sv)):
   for j in range(len(sv)):
    if i==j:continue
    weight=sign*np.prod(np.delete(sv,[i,j]))
    h+=weight*(np.outer(dd[:,i,i],dd[:,j,j])-np.outer(dd[:,i,j],dd[:,j,i]))
  return f,gradient,.5*(h+h.T)
 def evaluate(c,mu,curvature=True):
  focks=[];energy=0.;g=np.zeros(len(l));h=np.zeros((len(l),len(l)))
  for weight,o in [(2.,occ),(-1.,triplet)]:
   dm=np.array([(c*x)@c.T for x in o]);f,e=ctx.fock_and_energy(dm);energy+=weight*e;focks.append(f)
   qq=sum((c.T@ff@c)*x[None,:] for ff,x in zip(f,o));g+=weight*2*(qq[l,r]-qq[r,l])
   if curvature:
    state=SimpleNamespace(**vars(ctx.state));state.coefficients=c;state.alpha_occupations,state.beta_occupations=o
    kernel=ctx.density_response_kernel(state)
    h+=weight*orbital_residual_jacobian(c,o,f,kernel,ctx.polarization,l,r)
  a=qocc.T@square@c;ag=np.einsum('ij,ajk->aik',a,generators,optimize=True)
  d=a[:,alpha];first=ag[:,:,alpha]
  second=None
  if curvature:
   second=np.einsum('ain,bnj->abij',ag,generators[:,:,alpha],optimize=True)
   second=.5*(second+second.swapaxes(0,1))
  f,jj,hf=determinant_jet(d,first,second);constraint=np.array([f]);j=jj[None]
  lagg=g+j.T@mu;res=np.concatenate([lagg,constraint])
  if not curvature:return energy,res,j,None,focks,g
  h=.5*(h+h.T)+mu[0]*hf
  kkt=np.block([[h,j.T],[j,np.zeros((len(mu),len(mu)))]])
  return energy,res,j,kkt,focks,g
 _,_,j,_,_,g=evaluate(c,mu,curvature=False);mu=np.linalg.lstsq(j.T,-g,rcond=1e-12)[0]
 def retract(c,delta):
  return project(c@expm(np.einsum('a,aij->ij',delta,generators)))
 for iteration in range(201):
  energy,_,j,_,_,g=evaluate(c,np.zeros_like(mu),curvature=False)
  mu=np.linalg.lstsq(j.T,-g,rcond=1e-12)[0]
  energy,res,j,kkt,focks,g=evaluate(c,mu)
  error=float(np.max(abs(res)));z=null_space(j);values,vectors=np.linalg.eigh(z.T@kkt[:len(l),:len(l)]@z)
  index=int(np.sum(values<-1e-7))
  history.append(dict(iteration=iteration,energy=energy,residual=error,constraint=float(np.max(abs(res[len(l):]))),projected_index=index))
  np.savez_compressed(root/(label+'_latest.npz'),coefficients=c,multipliers=mu,history=json.dumps(history),reference_occupied=qocc)
  if error<1e-9 and index==0:break
  gradient=z.T@g;delta=-z@vectors@((vectors.T@gradient)/np.maximum(abs(values),1e-3))
  if error<1e-9 and index>0:delta=.03*z@vectors[:,0]
  norm=np.linalg.norm(delta)
  if norm>.15:delta*=.15/norm
  derivative=float(g@delta);accepted=False
  for fraction in [1.,.5,.25,.125,.0625,.03125,.015625]:
   trial=retract(c,fraction*delta);ee,_,_,_,_,_=evaluate(trial,mu,curvature=False)
   if ee<energy+1e-4*fraction*derivative:
    c=trial;accepted=True;break
  if not accepted:
   if error<1e-7 and index==0:break
   raise RuntimeError('constrained energy line search stalled: '+str(history[-1]))
 else:raise RuntimeError('constrained iteration limit: '+str(history[-1]))
 z=null_space(j);curvature=np.linalg.eigvalsh(z.T@kkt[:len(l),:len(l)]@z);index=int(np.sum(curvature<-1e-7))
 # Check the actual constrained Lagrangian curvature, not an unconstrained Hessian.
 direction=np.sin(np.arange(len(l))+.7);direction/=np.linalg.norm(direction);checks=[]
 for step in [2e-4,1e-4]:
  generator=np.einsum('a,aij->ij',step*direction,generators)
  _,minus,_,_,_,_=evaluate(c@expm(-generator),mu,curvature=False)
  _,plus,_,_,_,_=evaluate(c@expm(generator),mu,curvature=False)
  expected=kkt@np.concatenate([direction,np.zeros(len(mu))])
  checks.append(dict(step=step,hvp_error=float(np.max(abs((plus-minus)/(2*step)-expected)))))
 if checks[-1]['hvp_error']>1e-7:raise RuntimeError('KKT derivative check failed: '+str(checks))
 return dict(energy=energy,kkt_residual=error,projected_index=index,minimum_absolute_curvature=float(np.min(abs(curvature))),checks=checks,iterations=iteration),c


def main():
 root=BASE/'results/acetaldehyde/csf_orthogonal_roks';root.mkdir(exist_ok=True)
 if (root/'result.json').exists():return
 data=json.loads((BASE/'results/acetaldehyde/prepared.json').read_text());refs=[p.load_response_reference(x) for x in data['response_paths']]
 with np.load(BASE/'results/acetaldehyde/ground_replacement_minimum/accepted_geometry.npz') as d:relaxed=d['xyz'].copy()
 seeds=[];vertical=None
 for profile in ['full_response','qualified_spectral_response']:
  cp=Path('/home/enzo/MOUSE-roks-50-50-optimization-20261030/results/acetaldehyde')/profile/'seed.npz'
  with np.load(cp) as d:vertical=d['coordinates_angstrom'].copy()
  seeds.append(load_derivative_point(cp,request_hash='retained_vertical',coordinates_angstrom=vertical,require_gradient=False))
 calc=p.GFN2Calculator(data['atoms'],temperature_kelvin=0.);rows=[]
 anchors=[evaluate_restricted_ground(calc,vertical,template=s.reference_state,spin_reference=ref) for s,ref in zip(seeds,refs)]
 for geometry,xyz in [('vertical',vertical),('relaxed',relaxed)]:
  for i,(ref,old) in enumerate(zip(refs,seeds)):
   ctx=p.prepare_gfn2_deltascf(calc,xyz,orbital_reference=old.reference_state,spin_reference=ref)
   ground=evaluate_restricted_ground(calc,xyz,template=old.reference_state,spin_reference=ref)
   # Restore geometry after independent ground optimization and metric-chart assembly.
   ctx=p.prepare_gfn2_deltascf(calc,xyz,orbital_reference=old.reference_state,spin_reference=ref)
   u=TransportedOperators(ctx.state,old.reference_state.coordinates_bohr,np.eye(len(old.overlap))[None]).u
   square=metric_root(ctx.state.overlap,.5);inverse=metric_root(ctx.state.overlap,-.5);oldroot=metric_root(old.overlap,.5)
   initial=inverse@u@oldroot@old.common_coefficients
   for mode in ['frozen','current']:
    if geometry=='vertical' and mode=='current':continue
    qocc=(u@oldroot@anchors[i].coefficients[0][:,anchors[i].occupations[0]>.5] if mode=='frozen' else square@ground.coefficients[0][:,ground.occupations[0]>.5])
    try:
     result,c=solve(ctx,initial,old.occupations,qocc,root,f'{geometry}_{i}_{mode}')
     leak=float(abs(np.sqrt(2)*np.linalg.det(qocc.T@square@c[:,old.occupations[0]>.5])*np.linalg.det(qocc.T@square@c[:,old.occupations[1]>.5])))
     descriptor=replace(old,common_coefficients=c,overlap=ctx.state.overlap,reference_state=ctx.state)
     pair=active_orbital_character(old,descriptor,old.reference_state.coordinates_bohr)['transition']
     row=dict(geometry=geometry,component=i,projector=mode,gap_ev=(result['energy']-ground.energy_hartree)*27.211386245988,reference_ground_csf_overlap=leak,initial_pair_overlap=pair,**result)
    except Exception as exc:row=dict(geometry=geometry,component=i,projector=mode,error=str(exc))
    rows.append(row);print(json.dumps(row),flush=True);(root/'partial.json').write_text(json.dumps(rows,indent=2)+'\n')
 (root/'result.json').write_text(json.dumps(dict(rows=rows,nuclear_derivatives=False,no_fitted_offset=True),indent=2)+'\n')

if __name__=='__main__':
 with p.threadpool_limits(limits=2):main()
