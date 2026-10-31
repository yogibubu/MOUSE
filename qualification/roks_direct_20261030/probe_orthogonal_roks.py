"""Constrained, spin-pure ROKS feasibility with an exact virtual-particle constraint.

The excited particle lies in the virtual space of a supplied closed-shell
reference. All other orbitals relax. A KKT stationary solve uses analytic
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
 q=square@initial;vector=q[:,particle]-qocc@(qocc.T@q[:,particle]);norm=np.linalg.norm(vector)
 if norm<.5:raise ValueError('initial particle does not map to the reference virtual space')
 vector/=norm;b=q[:,others]-np.outer(vector,vector@q[:,others]);u,_,vt=np.linalg.svd(b,full_matrices=False)
 q[:,others]=u@vt;q[:,particle]=vector;c=metric_root(ctx.state.overlap,-.5)@q
 mu=np.zeros(qocc.shape[1]);history=[]
 def evaluate(c,mu,curvature=True):
  focks=[];energy=0.;g=np.zeros(len(l));h=np.zeros((len(l),len(l)))
  for weight,o in [(2.,occ),(-1.,triplet)]:
   dm=np.array([(c*x)@c.T for x in o]);f,e=ctx.fock_and_energy(dm);energy+=weight*e;focks.append(f)
   qq=sum((c.T@ff@c)*x[None,:] for ff,x in zip(f,o));g+=weight*2*(qq[l,r]-qq[r,l])
   if curvature:
    state=SimpleNamespace(**vars(ctx.state));state.coefficients=c;state.alpha_occupations,state.beta_occupations=o
    kernel=ctx.density_response_kernel(state)
    h+=weight*orbital_residual_jacobian(c,o,f,kernel,ctx.polarization,l,r)
  a=qocc.T@square@c;constraint=a[:,particle];ag=np.einsum('ij,ajk->aik',a,generators,optimize=True);j=ag[:,:,particle].T
  lagg=g+j.T@mu;res=np.concatenate([lagg,constraint])
  if not curvature:return energy,res,j,None,focks,g
  second=np.einsum('ain,bn->iab',ag,generators[:,:,particle],optimize=True);second=.5*(second+second.transpose(0,2,1))
  h=.5*(h+h.T)+np.einsum('i,iab->ab',mu,second)
  kkt=np.block([[h,j.T],[j,np.zeros((len(mu),len(mu)))]])
  return energy,res,j,kkt,focks,g
 _,_,j,_,_,g=evaluate(c,mu,curvature=False);mu=np.linalg.lstsq(j.T,-g,rcond=1e-12)[0]
 for iteration in range(101):
  energy,res,j,kkt,focks,g=evaluate(c,mu)
  error=float(np.max(abs(res)));history.append(dict(iteration=iteration,energy=energy,residual=error,constraint=float(np.max(abs(res[len(l):])))))
  np.savez_compressed(root/(label+'_latest.npz'),coefficients=c,multipliers=mu,history=json.dumps(history),reference_occupied=qocc)
  if error<1e-9:break
  step=np.linalg.lstsq(kkt,-res,rcond=1e-11)[0];norm=np.linalg.norm(step[:len(l)])
  if norm>.15:step*=.15/norm
  for fraction in [1.,.5,.25,.125,.0625,.03125]:
   generator=np.einsum('a,aij->ij',fraction*step[:len(l)],generators);trial=c@expm(generator);mtrial=mu+fraction*step[len(l):]
   _,rr,_,_,_,_=evaluate(trial,mtrial,curvature=False)
   if np.linalg.norm(rr)<np.linalg.norm(res):c,mu=trial,mtrial;break
  else:raise RuntimeError('KKT residual line search stalled: '+str(history[-1]))
 else:raise RuntimeError('KKT iteration limit: '+str(history[-1]))
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
 root=BASE/'results/acetaldehyde/orthogonal_roks';root.mkdir(exist_ok=True)
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
     particle=int(np.flatnonzero(old.occupations[0]-old.occupations[1]==1)[0]);leak=float(np.linalg.norm(qocc.T@square@c[:,particle]))
     descriptor=replace(old,common_coefficients=c,overlap=ctx.state.overlap,reference_state=ctx.state)
     pair=active_orbital_character(old,descriptor,old.reference_state.coordinates_bohr)['transition']
     row=dict(geometry=geometry,component=i,projector=mode,gap_ev=(result['energy']-ground.energy_hartree)*27.211386245988,reference_ground_particle_leak=leak,initial_pair_overlap=pair,**result)
    except Exception as exc:row=dict(geometry=geometry,component=i,projector=mode,error=str(exc))
    rows.append(row);print(json.dumps(row),flush=True);(root/'partial.json').write_text(json.dumps(rows,indent=2)+'\n')
 (root/'result.json').write_text(json.dumps(dict(rows=rows,nuclear_derivatives=False,no_fitted_offset=True),indent=2)+'\n')

if __name__=='__main__':
 with p.threadpool_limits(limits=2):main()
