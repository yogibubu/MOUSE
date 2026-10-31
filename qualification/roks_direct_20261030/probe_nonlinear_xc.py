"""Remote vertical-only full semilocal XC remainder, with analytic orbital Hessian.

Adds E_xc[core+L P L^T] minus its complete second-order Taylor expansion
at the SAME embedded native reference used by the frozen response. Thus no
parameter, orbital-energy offset, lambda change or TBE is introduced. Nuclear
derivatives are deliberately NOT exposed by this feasibility probe.
"""
import importlib.util,json,time,sys,hashlib
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
from scipy.linalg import expm,solve_sylvester
BASE=Path('/home/enzo/MOUSE-roks-direct-20261030');OLD=Path('/home/enzo/MOUSE-roks-independent-20261030')
spec=importlib.util.spec_from_file_location('p',BASE/'source/qualification/probe.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
from pyscf import dft,scf,lib
from matrix_gfn2.orbital_saddle import generalized_mode_following
from matrix_gfn2.deltascf_derivatives import load_derivative_point,save_derivative_point
from matrix_gfn2.roks import ROKSSingletResult
from matrix_gfn2.roks_tracking import active_orbital_character
np=p.np;lib.num_threads(2)

class XCResidual:
 def __init__(self,mol,lift,core,state,reference,cache_path):
  operators=reference.operators;s=state.overlap
  self.b=.5*(s@operators+operators@s)
  fingerprint=hashlib.sha256(lift.tobytes()+core.tobytes()+reference.fingerprint.encode()+b'PBE0.grid3.deriv1').hexdigest()
  if cache_path.exists():
   with np.load(cache_path) as d:
    assert str(d['fingerprint'])==fingerprint
    self.features=d['features'].copy();self.core=d['core'].copy();self.w=d['weights'].copy()
  else:
   ni=dft.numint.NumInt();grid=dft.gen_grid.Grids(mol);grid.level=3;grid.build(with_non0tab=True)
   dual=np.array([2*solve_sylvester(s,s,a) for a in operators]);features=[];cores=[];weights=[]
   for ao,mask,w,coords in ni.block_loop(mol,grid,mol.nao_nr(),deriv=1,max_memory=1000):
    phi=ao@lift;f=np.empty((4,len(operators),len(w)))
    f[0]=np.einsum('gi,aij,gj->ag',phi[0],dual,phi[0],optimize=True)
    for k in range(1,4):f[k]=2*np.einsum('gi,aij,gj->ag',phi[k],dual,phi[0],optimize=True)
    features.append(f);cores.append(ni.eval_rho(mol,ao,core@core.T,mask,'GGA',hermi=1));weights.append(w)
   self.features=np.concatenate(features,axis=2);self.core=np.concatenate(cores,axis=1);self.w=np.concatenate(weights)
   temporary=cache_path.with_suffix('.tmp')
   with temporary.open('wb') as stream:np.savez_compressed(stream,fingerprint=fingerprint,features=self.features,core=self.core,weights=self.w)
   temporary.replace(cache_path)
  self.q0=np.array([reference.pair_reference_moments]*2)*.5
  self.rho0=self.rho(self.q0);self.e0,self.v0,self.k0=self.xc(self.rho0);self.cache={};self.strength=1.
 def rho(self,q):return self.core[None]+np.einsum('sA,iAg->sig',q,self.features,optimize=True)
 def xc(self,rho):
  if np.min(rho[:,0]) < -1e-10:raise ValueError('negative embedded spin density')
  exc,v,k=dft.numint.NumInt().eval_xc_eff('PBE0',rho,deriv=2,xctype='GGA')[:3]
  return float(np.sum(self.w*np.sum(rho[:,0],axis=0)*exc)),v,k
 def moments(self,dp):return np.einsum('aij,...sij->...sa',self.b,dp,optimize=True)
 def evaluate(self,dm):
  q=self.moments(dm)
  if 'q' in self.cache and np.array_equal(q,self.cache['q']):return self.strength*self.cache['f'],self.strength*self.cache['e']
  rho=self.rho(q);e,v,k=self.xc(rho);drho=rho-self.rho0
  linear=np.einsum('sig,sig,g->',self.v0,drho,self.w,optimize=True)
  response0=np.einsum('sitjg,tjg->sig',self.k0,drho,optimize=True)
  quadratic=.5*np.einsum('sig,sig,g->',drho,response0,self.w,optimize=True)
  field=np.einsum('sig,iag,g->sa',v-self.v0-response0,self.features,self.w,optimize=True)
  f=np.einsum('sa,aij->sij',field,self.b,optimize=True);e=e-self.e0-linear-quadratic
  self.cache=dict(q=q.copy(),f=f,e=e,k=k-self.k0)
  return self.strength*f,self.strength*float(e)
 def response(self,dm,directions):
  self.evaluate(dm)
  dq=self.moments(directions)
  drho=np.einsum('bsa,iag->bsig',dq,self.features,optimize=True)
  dv=np.einsum('sitjg,btjg->bsig',self.cache['k'],drho,optimize=True)
  field=np.einsum('bsig,iag,g->bsa',dv,self.features,self.w,optimize=True)
  return self.strength*np.einsum('bsa,aij->bsij',field,self.b,optimize=True)


def solve(ctx,residual,c0,occ,index,checkpoint):
 triplet=np.array([np.maximum(*occ),np.minimum(*occ)])
 components=[(1.,occ)] if np.array_equal(*occ) else [(2.,occ),(-1.,triplet)]
 left,right=np.triu_indices(len(c0),1);labels=np.vstack([o for _,o in components]);active=np.any(labels[:,left]!=labels[:,right],axis=0);left,right=left[active],right[active]
 if checkpoint.exists():
  with np.load(checkpoint) as d:
   assert int(d['index'])==index;c0=d['coefficients'].copy()
 cache={}
 def objective(c,curvature=True):
  if 'c' not in cache or not np.array_equal(c,cache['c']):
   values=[];energy=0.;g=np.zeros(len(left))
   for weight,o in components:
    dm=np.array([(c*x)@c.T for x in o]);f,e=ctx.fock_and_energy(dm);df,de=residual.evaluate(dm);f=f+df;e+=de
    q=sum((c.T@ff@c)*x[None,:] for ff,x in zip(f,o));g+=weight*2*(q[left,right]-q[right,left]);energy+=weight*e;values.append((dm,f))
   cache.clear();cache.update(c=c.copy(),energy=energy,g=g,values=values)
  if curvature and 'h' not in cache:
   dc=np.zeros((len(left),len(c),len(c)))
   for b,(i,j) in enumerate(zip(left,right)):dc[b,:,j]=c[:,i];dc[b,:,i]=-c[:,j]
   h=np.zeros((len(left),len(left)))
   for (weight,o),(dm,f) in zip(components,cache['values']):
    dp=np.array([(dc*x)@c.T for x in o]).transpose(1,0,2,3);dp+=dp.transpose(0,1,3,2)
    state=SimpleNamespace(**vars(ctx.state));state.coefficients=c;state.alpha_occupations,state.beta_occupations=o
    kernel=ctx.density_response_kernel(state)
    df=np.array([kernel(d.sum(axis=0))[None]+ctx.polarization.evaluate(d)[0] for d in dp])+residual.response(dm,dp)
    for spin in range(2):
     mo=dc.transpose(0,2,1)@f[spin]@c+c.T@f[spin]@dc+c.T@df[:,spin]@c
     h+=weight*(2*mo[:,left,right]*(o[spin,right]-o[spin,left])).T
   cache['h']=.5*(h+h.T)
  return cache['energy'],cache['g'],cache.get('h')
 def retract(c,step):
  k=np.zeros_like(c);k[left,right]=step;k[right,left]=-step;return c@expm(k)
 def save(r):
  np.savez_compressed(checkpoint,coefficients=r.point,index=index,residual=np.max(abs(r.gradient)),energy=r.energy,history=json.dumps(r.history))
 solved=generalized_mode_following(objective,retract,c0,saddle_index=index,gradient_tolerance=1e-9,maximum_iterations=80,maximum_step=.15,checkpoint=save)
 e,g,_=objective(solved.point,curvature=False)
 if not solved.converged:raise RuntimeError(str(solved.history[-1]))
 _,g,h=objective(solved.point,curvature=True);direction=np.sin(np.arange(len(g))+.3);direction/=np.linalg.norm(direction)
 checks=[]
 for step in [2e-4,1e-4]:
  em,gm,_=objective(retract(solved.point,-step*direction),curvature=False)
  ep,gp,_=objective(retract(solved.point,step*direction),curvature=False)
  checks.append(dict(step=step,gradient_error=float(abs((ep-em)/(2*step)-g@direction)),hvp_error=float(np.max(abs((gp-gm)/(2*step)-h@direction)))))
 objective(solved.point,curvature=False);cache['derivative_checks']=checks
 if any(x['gradient_error']>1e-7 or x['hvp_error']>1e-5 for x in checks):raise RuntimeError('orbital derivative validation failed: '+str(checks))
 return solved,cache


def main(name):
 root=BASE/'results'/name/'nonlinear_xc';root.mkdir(parents=True,exist_ok=True)
 if (root/'result.json').exists():return
 cp=Path('/home/enzo/MOUSE-roks-50-50-optimization-20261030/results')/name/'full_response/seed.npz'
 if cp.exists():
  with np.load(cp) as d:xyz=d['coordinates_angstrom'].copy()
  old=load_derivative_point(cp,request_hash='retained_vertical',coordinates_angstrom=xyz,require_gradient=False)
  data=json.loads((BASE/'results'/name/'prepared.json').read_text());refs=[p.load_response_reference(x) for x in data['response_paths']]
 else:
  record=json.loads((OLD/'results/full_response'/name/'n_pi/result.json').read_text());assert record['qualified']
  cp=Path(record['checkpoint'])
  with np.load(cp) as d:xyz=d['coordinates_angstrom'].copy()
  old=load_derivative_point(cp,request_hash=record['request_hash'],coordinates_angstrom=xyz,require_gradient=False)
  state=old.reference_state;data=dict(atoms=[{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in state.atomic_numbers]);refs=[]
  calc0=p.GFN2Calculator(data['atoms'],temperature_kelvin=0.);ctx0=p.prepare_gfn2_deltascf(calc0,xyz,orbital_reference=state)
  from matrix_gfn2 import FrozenNonlocalResponseReference,save_response_reference
  for profile in ['full_response','qualified_spectral_response']:
   path=OLD/'results'/profile/name/'model.npz'
   with np.load(path) as d:model={k:d[k].copy() for k in ['operators','charge','spin','q0','linear']}
   ref=FrozenNonlocalResponseReference.from_vertical_model(state,ctx0.kernel.partition,model,hashlib.sha256(path.read_bytes()).hexdigest())
   refs.append(ref);save_response_reference(root/(profile+'.npz'),ref)
 state=old.reference_state
 casepath=OLD/'prepared_B'/name/'case.json'
 if casepath.exists():folder=Path(json.loads(casepath.read_text())['reference'])
 elif name=='nitroxyl':folder=Path('/home/enzo/MATRIX-runs/tda-dft-independent-20261022-v1/docs/reports/independent_dft_tda_20261022/results/nitroxyl_pbe0')
 else:raise ValueError('explicit source reference required for '+name)
 embedding=OLD/'results/full_response'/name/'embedding.npz'
 if not embedding.exists():embedding=Path('/home/enzo/MOUSE-roks-repair-20261030/results/spin_kernel')/name/'kernel.npz'
 with np.load(embedding) as d:lift=d['embedding'].copy()
 with np.load(folder/'reference.npz') as d:coeff=d['coefficients'];occ=d['occupations']
 nc=int(np.sum(old.core_electrons)//2);core=coeff[:,np.flatnonzero(occ==2)[:nc]];mol=scf.chkfile.load_mol(str(folder/'backend.chk'))
 extra=XCResidual(mol,lift,core,state,refs[0],root/'grid_features.npz');dm0=np.array([(state.coefficients*o)@state.coefficients.T for o in [state.alpha_occupations,state.beta_occupations]])
 zero_f,zero_e=extra.evaluate(dm0);assert abs(zero_e)<1e-10 and np.max(abs(zero_f))<1e-10
 rng=np.random.default_rng(741);direction=rng.normal(size=dm0.shape);direction=.5*(direction+direction.transpose(0,2,1))
 assert np.max(abs(extra.response(dm0,direction[None])))<1e-9
 calc=p.GFN2Calculator(data['atoms'],temperature_kelvin=0.);rows=[];start=time.monotonic()
 for i,ref in enumerate(refs):
  context=p.prepare_gfn2_deltascf(calc,xyz,orbital_reference=state,spin_reference=ref)
  ground_occ=np.array([state.alpha_occupations,state.beta_occupations])
  solved0,cache0=solve(context,extra,state.coefficients,ground_occ,0,root/f'ground_{i}_latest.npz')
  solved,cache=solve(context,extra,old.common_coefficients,old.occupations,old.orbital_saddle_index,root/f'excited_{i}_latest.npz')
  result=ROKSSingletResult(solved.point,old.occupations,old.triplet_occupations,state.overlap,cache['values'][0][1],cache['values'][1][1],solved.energy,float(np.max(abs(solved.gradient))),solved.iterations,True,ground_energy_hartree=solved0.energy,reference_state=state,core_electrons=old.core_electrons,orbital_saddle_index=solved.observed_index,target_orbital_saddle_index=solved.target_index,spin_reference_fingerprint='nonlinear_xc_vertical:'+ref.fingerprint)
  pair=active_orbital_character(old,result,state.coordinates_bohr,allow_different_response=True)['transition']
  save_derivative_point(root/f'excited_{i}.npz',result,request_hash='nonlinear_xc_vertical_v1',coordinates_angstrom=xyz)
  row=dict(component=i,orbital_derivative_checks=cache['derivative_checks'],ground_orbital_derivative_checks=cache0['derivative_checks'],excitation_ev=result.excitation_energy_hartree*27.211386245988,index=result.orbital_saddle_index,initial_pair_overlap=pair,residual=result.orbital_residual,seconds=time.monotonic()-start)
  rows.append(row);print(json.dumps(row),flush=True)
 (root/'result.json').write_text(json.dumps(dict(molecule=name,rows=rows,mean_excitation_ev=.5*sum(r['excitation_ev'] for r in rows),nuclear_derivatives=False,lambda_exchange=.9,fitted_offset=0.),indent=2)+'\n')

if __name__=='__main__':
 with p.threadpool_limits(limits=2):main(sys.argv[1])
