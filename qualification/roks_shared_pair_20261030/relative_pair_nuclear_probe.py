"""Remote analytic KKT nuclear Hessian feasibility, independently differentiated."""
import probe_relative_pair as p
import json,time
from pathlib import Path
from relative_pair_objective import RelativePairObjective
from matrix_gfn2.deltascf_analytic_hessian import _fixed_density_geometry,_response_blocks
np=p.np
BASE=Path('/home/enzo/MOUSE-roks-shared-pair-20261030')

def stationary(obj,point,mu):
 for it in range(41):
  e,res,kkt,j=obj.evaluate(point,mu);err=float(np.max(abs(res)))
  if err<1e-10:return e,point,mu,kkt,j
  step=np.linalg.lstsq(kkt,-res,rcond=1e-12)[0];norm=np.linalg.norm(step[:obj.dimension])
  if norm>.1:step*=.1/norm
  for scale in [1.,.5,.25,.125,.0625,.03125,.015625]:
   c=obj.retract(point,scale*step[:obj.dimension]);m=mu+scale*step[obj.dimension:];_,rr,_,_=obj.evaluate(c,m,curvature=False)
   if np.linalg.norm(rr)<np.linalg.norm(res):point,mu=c,m;break
  else:raise RuntimeError('KKT geometry continuation stalled '+str(err))
 raise RuntimeError('KKT geometry continuation iteration limit')

def evaluate(calc,xyz,refs,parents,coeff,mu,old_s,target,hessian=False):
 ctxs=[p.prepare_gfn2_deltascf(calc,xyz,orbital_reference=parent.reference_state,spin_reference=ref) for parent,ref in zip(parents,refs)]
 s=ctxs[0].state.overlap;point=np.array([p.root_metric(s,-.5)@p.root_metric(old_s,.5)@c for c in coeff])
 obj=RelativePairObjective(ctxs,parents[0].occupations,target);e,point,mu,kkt,j=stationary(obj,point,mu.copy());obj.evaluate(point,mu)
 grad=np.zeros_like(xyz);nr=xyz.size;hrr=np.zeros((nr,nr));b=np.zeros((obj.dimension,nr));native_h=np.zeros((obj.dimension,obj.dimension))
 for i,(c,ctx,ff) in enumerate(zip(point,ctxs,obj.cache['focks'])):
  dm=np.array([(c*o)@c.T for o in obj.occ]);dmt=np.array([(c*o)@c.T for o in obj.triplet])
  q=sum(2*(c.T@f@c)*o[None,:] for f,o in zip(ff[0],obj.occ))-sum((c.T@f@c)*o[None,:] for f,o in zip(ff[1],obj.triplet))
  weighted=c@(.5*(q+q.T))@c.T
  grad+=.5*(2*ctx.analytic_gradient(dm,.5*weighted)-ctx.analytic_gradient(dmt,np.zeros_like(weighted)))
  if hessian:
   kernel,ds,df,d2s,err=_fixed_density_geometry(ctx,dm)
   if ctx.charge_correction is not None:
    charge=ctx.charge_correction;dfield,efield=charge.geometry(dm.sum(axis=0),ds,d2s);df,err=df+dfield,err+efield;base=kernel;kernel=lambda dp,base=base,charge=charge:base(dp)+charge.response(dp)
   rr,kk,kr=_response_blocks([c],[(2.,obj.occ,ff[0]),(-1.,obj.triplet,ff[1])],ds,d2s,[(df,err)]*2,[kernel]*2,ctx.polarization,orbital_pairs=(obj.l,obj.r))
   sl=slice(i*obj.m,(i+1)*obj.m);hrr+=.5*rr;b[sl]=.5*kr;native_h[sl,sl]=.5*kk
 if not hessian:return e,grad,point,mu,s
 error=float(np.max(abs(native_h-obj.cache['h'])));assert error<1e-8,error
 # The cross-parent metric overlaps are exactly R-independent in the common
 # metric-preserving Pulay chart. All explicit nuclear constraint jets vanish.
 augmented=np.vstack([b,np.zeros((2,nr))]);response=np.linalg.solve(kkt,-augmented);h=hrr+augmented.T@response
 return e,grad,point,mu,s,.5*(h+h.T),dict(native_orbital_hessian_error=error,kkt_response_residual=float(np.max(abs(kkt@response+augmented))))

def run():
 root=BASE/'results/formamide/relative_active_pair';data=json.loads(Path('/home/enzo/MOUSE-roks-local-20261030/results/formamide/prepared.json').read_text())
 accepted=Path('/home/enzo/MOUSE-roks-local-20261030/results/formamide/optimization/roks_50_50_accepted');manifest=json.loads((accepted/'accepted.json').read_text());parents=[p.load(accepted/row['path']) for row in manifest['reference']];refs=[p.load_response_reference(path) for path in data['response_paths']]
 calc=p.GFN2Calculator(data['atoms'],temperature_kelvin=0.);calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('SCC invoked'))
 with np.load(root/'latest.npz') as d:point=d['coefficients'].copy();mu=d['multipliers'].copy();target=d['target'].copy()
 xyz=parents[0].reference_state.coordinates_bohr/p.ANGSTROM_TO_BOHR;s=parents[0].overlap
 start=time.monotonic();central=evaluate(calc,xyz,refs,parents,point,mu,s,target,hessian=True);e,g,point,mu,s,h,diagnostics=central
 direction=np.sin(np.arange(xyz.size)+.31).reshape(xyz.shape);direction-=direction.mean(axis=0);direction/=np.linalg.norm(direction);checks=[]
 for step in [2e-4,1e-4]:
  a=evaluate(calc,xyz-step/p.ANGSTROM_TO_BOHR*direction,refs,parents,point,mu,s,target);b=evaluate(calc,xyz+step/p.ANGSTROM_TO_BOHR*direction,refs,parents,point,mu,s,target)
  checks.append(dict(step=step,gradient_error=float(abs((b[0]-a[0])/(2*step)-np.sum(g*direction))),hvp_error=float(np.max(abs((b[1]-a[1]).ravel()/(2*step)-h@direction.ravel())))))
 out=dict(checks=checks,diagnostics=diagnostics,seconds=time.monotonic()-start,passed=all(r['gradient_error']<1e-7 and r['hvp_error']<1e-6 for r in checks))
 (root/'nuclear_validation.json').write_text(json.dumps(out,indent=2)+'\n');np.savez_compressed(root/'nuclear_analytic.npz',xyz=xyz,gradient=g,hessian=h);print(json.dumps(out),flush=True)
if __name__=='__main__':
 with p.threadpool_limits(limits=2):run()
