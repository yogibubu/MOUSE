"""Oracle-only common-orbital PBE0 singlet relaxation diagnostic.

The ROKS objective is 2 E_UKS(Ms=0) - E_UKS(Ms=1) on ONE set of
orbitals. Full AO space, no frozen cores, no fitted scaling. Newton solves
orbital stationarity (including saddles), with analytic UKS response HVPs.
This orbital Hessian is not a nuclear Hessian of the hybrid model.
"""
from remote_bootstrap import HOST
from pathlib import Path
import argparse,json,time,traceback,hashlib
import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator,gmres
from pyscf import dft,scf,lib
from pyscf.scf import _response_functions  # register gen_response
from frozen_relaxation import digest,write,key,EV
from matrix_apoc.deltascf import (DeterminantOrbitals,DeltaSCFState,
                                SpinAdaptedSinglet,singlet_csf_overlap)

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'dft_relaxation'
PROTOCOL=dict(schema='common-orbital-dft-relaxation-v1',host_required='oracle.sns.it',
 xc='PBE0',grid_level=3,small_rho_cutoff=0.,threads=2,max_memory_mb=20000,
 orbital_tolerance=1e-7,maximum_iterations=45,trust_radius=.35,
 minimum_seed_csf_overlap=.70,minimum_step_csf_overlap=.90,
 gmres_rtol=.08,gmres_restart=24,gmres_maxiter=2,
 orbital_space='full AO; all core and valence orbitals may relax',
 objective='2*UKS(mixed)-UKS(triplet), common orbitals',
 solver='analytic-response Newton-GMRES with residual line search',
 cases=['uracil','thymine'],states=['n_pi','pi_pi'],parameters_fitted=0)
lib.num_threads(PROTOCOL['threads'])

class Objective:
 def __init__(self,case,state):
  self.case=case;self.state=state
  source=ROOT/'frozen_relaxation'/case
  self.previous=json.loads((source/'result.json').read_text())
  self.row=next(r for r in self.previous['records'] if r['id']==state)
  plan=json.loads((ROOT/'frozen_relaxation/plan.json').read_text())
  folder=Path(next(c for c in plan['cases'] if c['name']==case)['reference'])
  self.request=key(dict(protocol=PROTOCOL,reference_sha256=digest(folder/'reference.npz'),
   checkpoint_sha256=digest(folder/'backend.chk'),state=state,
   hole=self.row['dft_hole'],particle=self.row['dft_particle']))
  ref=np.load(folder/'reference.npz');self.ref={k:ref[k] for k in ref.files};ref.close()
  self.C0=self.ref['coefficients'];self.S=self.ref['overlap'];self.n=len(self.S)
  self.occ=np.array([self.ref['occupations']/2]*2)
  self.occ[0,self.row['dft_hole']]=0;self.occ[0,self.row['dft_particle']]=1
  self.trip=np.array([np.maximum(*self.occ),np.minimum(*self.occ)])
  i,j=np.triu_indices(self.n,1);labels=np.vstack((self.occ,self.trip))
  active=np.any(labels[:,i]!=labels[:,j],axis=0);self.i=i[active];self.j=j[active];self.size=len(self.i)
  self.mol=scf.chkfile.load_mol(str(folder/'backend.chk'));self.mol.max_memory=PROTOCOL['max_memory_mb']
  assert np.max(abs(self.mol.intor('int1e_ovlp')-self.S))<1e-9
  self.mf=dft.UKS(self.mol);self.mf.xc=PROTOCOL['xc'];self.mf.verbose=0
  self.mf.max_memory=PROTOCOL['max_memory_mb'];self.mf.small_rho_cutoff=0
  self.mf.grids.level=PROTOCOL['grid_level'];self.mf.grids.build(with_non0tab=True)
  self.h=self.mf.get_hcore()
  # One shared exact in-core ERI tensor; no density-fitting approximation.
  self.mf._eri=self.mol.intor('int2e',aosym='s8')
  self.ground=self.previous['ground']['energy_hartree'];self.calls=0
 def K(self,x):
  k=np.zeros((self.n,self.n));k[self.i,self.j]=x;k[self.j,self.i]=-x;return k
 def rotate(self,c,x):return c@expm(self.K(x))
 def csf(self,c,energy=0.):
  w=DeterminantOrbitals(self.ref['atomic_numbers'],self.ref['coordinates_bohr'],self.S,
   (c,c),tuple(self.occ),np.zeros(len(self.ref['atomic_numbers']),dtype=int),
   charge=self.mol.charge,multiplicity=1,total_energy_hartree=energy,model='PBE0/aug-cc-pVDZ ROKS')
  return SpinAdaptedSinglet(1,DeltaSCFState(1,w),energy)
 def overlap(self,c,d):return singlet_csf_overlap(self.csf(c),self.csf(d),self.S)[0]
 def evaluate(self,c):
  self.calls+=1;fs=[];es=[];dens=[]
  for occ in (self.occ,self.trip):
   dm=np.array([(c*o)@c.T for o in occ]);v=self.mf.get_veff(self.mol,dm)
   es.append(float(self.mf.energy_tot(dm=dm,h1e=self.h,vhf=v)))
   fs.append(np.array([c.T@(self.h+f)@c for f in v]));dens.append(dm)
  q=sum(2*f*o[None,:] for f,o in zip(fs[0],self.occ))
  q-=sum(f*o[None,:] for f,o in zip(fs[1],self.trip))
  g=2*(q[self.i,self.j]-q[self.j,self.i])
  return dict(c=c,energy=2*es[0]-es[1],mixed_energy=es[0],triplet_energy=es[1],
              g=g,f=fs,dm=dens)
 def response(self,v):
  c=v['c'];responses=[]
  for occ in (self.occ,self.trip):
   responses.append(self.mf.gen_response(mo_coeff=np.array([c,c]),mo_occ=occ,hermi=1))
  calls=[0]
  def hv(x):
   calls[0]+=1;k=self.K(x);dq=np.zeros((self.n,self.n))
   for factor,occ,f,respond in zip((2.,-1.),(self.occ,self.trip),v['f'],responses):
    dd=np.array([c@(k*o[None,:]-o[:,None]*k)@c.T for o in occ])
    df=respond(dd)
    for o,f0,df0 in zip(occ,f,df):
     dfmo=c.T@df0@c+f0@k-k@f0
     dq+=factor*dfmo*o[None,:]
   return 2*(dq[self.i,self.j]-dq[self.j,self.i])
  diag=np.zeros(self.size)
  for factor,occ,f in zip((2.,-1.),(self.occ,self.trip),v['f']):
   for o,f0 in zip(occ,f):
    d=np.diag(f0);diag+=2*factor*(d[self.i]-d[self.j])*(o[self.j]-o[self.i])
  diag=np.where(abs(diag)<.02,np.where(diag<0,-.02,.02),diag)
  return hv,diag,calls

def run(case,state,check_only=False):
 out=OUT/case/state;out.mkdir(parents=True,exist_ok=True)
 final=out/'result.json'
 if final.exists():
  old=json.loads(final.read_text())
  if old.get('converged') and not check_only:
   print('REUSE',case,state,old,flush=True);return old
 obj=Objective(case,state);cp=out/'latest.npz';histpath=out/'history.json';hist=[]
 c=obj.C0.copy()
 if cp.exists():
  with np.load(cp) as f:
   assert str(f['request_hash'])==obj.request
   c=f['coefficients'].copy()
  hist=json.loads(histpath.read_text()) if histpath.exists() else []
 v=obj.evaluate(c);print('START',HOST,case,state,'nao',obj.n,'rotations',obj.size,'E',v['energy'],flush=True)
 audit=out/'derivative_audit.json'
 if not audit.exists():
  rng=np.random.default_rng(20261030);direction=rng.normal(size=obj.size);direction/=np.linalg.norm(direction)
  h=2e-5;vp=obj.evaluate(obj.rotate(c,h*direction));vm=obj.evaluate(obj.rotate(c,-h*direction))
  hv,diag,calls=obj.response(v);analytic=hv(direction);numeric=(vp['g']-vm['g'])/(2*h)
  data=dict(host=HOST,request_hash=obj.request,step=h,
   energy_direction_error=abs((vp['energy']-vm['energy'])/(2*h)-v['g']@direction),
   response_max_error=float(np.max(abs(analytic-numeric))),
   response_relative_error=float(np.linalg.norm(analytic-numeric)/np.linalg.norm(numeric)))
  data['passed']=data['energy_direction_error']<2e-7 and data['response_max_error']<2e-6 and data['response_relative_error']<2e-5
  write(audit,data);print('DERIVATIVES',data,flush=True)
  if not data['passed']:raise ValueError('analytic orbital derivatives failed directional check')
 else:
  a=json.loads(audit.read_text());assert a['passed'] and a['request_hash']==obj.request
 if check_only:return
 started=time.perf_counter()
 for iteration in range(len(hist),PROTOCOL['maximum_iterations']):
  error=float(np.max(abs(v['g'])));norm=float(np.linalg.norm(v['g']))
  overlap=obj.overlap(obj.C0,c)
  record=dict(iteration=iteration,energy_hartree=v['energy'],excitation_ev=(v['energy']-obj.ground)*EV,
   orbital_max_residual=error,orbital_residual_norm=norm,seed_csf_overlap=overlap,
   seconds_current_run=time.perf_counter()-started,fock_pairs_current_run=obj.calls)
  hist.append(record);write(histpath,hist)
  tmp=cp.with_suffix('.tmp.npz');np.savez_compressed(tmp,request_hash=obj.request,coefficients=c,
   occupations=obj.occ,energy_hartree=v['energy'],gradient=v['g']);tmp.replace(cp)
  print('ITER',case,state,record,flush=True)
  if error<=PROTOCOL['orbital_tolerance']:break
  hv,diag,hcalls=obj.response(v)
  a=LinearOperator((obj.size,obj.size),matvec=hv,dtype=float)
  pre=LinearOperator(a.shape,matvec=lambda x:x/diag,dtype=float)
  step,info=gmres(a,-v['g'],M=pre,rtol=PROTOCOL['gmres_rtol'],atol=0,
   restart=PROTOCOL['gmres_restart'],maxiter=PROTOCOL['gmres_maxiter'])
  sn=np.linalg.norm(step)
  if not np.isfinite(sn) or sn==0:raise ValueError('invalid Newton direction')
  step*=min(1.,PROTOCOL['trust_radius']/sn)
  accepted=False
  for scale in (1.,.5,.25,.125,.0625):
   trial=obj.rotate(c,scale*step)
   if obj.overlap(obj.C0,trial)<PROTOCOL['minimum_seed_csf_overlap'] or obj.overlap(c,trial)<PROTOCOL['minimum_step_csf_overlap']:continue
   trialv=obj.evaluate(trial)
   if np.linalg.norm(trialv['g'])<norm*(1-1e-4*scale):
    c=trial;v=trialv;accepted=True;break
  record.update(response_products=hcalls[0],gmres_info=int(info),newton_step_norm=float(sn),accepted_scale=scale if accepted else None)
  write(histpath,hist)
  if not accepted:raise RuntimeError('Newton residual line search failed; latest checkpoint retained')
 error=float(np.max(abs(v['g'])));qualified=error<=PROTOCOL['orbital_tolerance'] and obj.overlap(obj.C0,c)>=PROTOCOL['minimum_seed_csf_overlap']
 dft_ev=(v['energy']-obj.ground)*EV;dft_relax=dft_ev-obj.row['dft_frozen_ev']
 result=dict(host=HOST,request_hash=obj.request,molecule=case,state=state,protocol=PROTOCOL,
  converged=qualified,orbital_max_residual=error,seed_csf_overlap=obj.overlap(obj.C0,c),spin_squared_csf=0.,
  energy_hartree=v['energy'],mixed_energy_hartree=v['mixed_energy'],triplet_energy_hartree=v['triplet_energy'],
  ground_hartree=obj.ground,dft_frozen_ev=obj.row['dft_frozen_ev'],dft_relaxed_ev=dft_ev,
  dft_relaxation_ev=dft_relax,se_relaxation_ev=obj.row['se_relaxation_ev'],
  relaxation_mismatch_ev=obj.row['se_relaxation_ev']-dft_relax,
  tbe_ev=obj.row['tbe_ev'],dft_error_ev=dft_ev-obj.row['tbe_ev'],hybrid_ev=obj.row['hybrid_ev'],
  hybrid_error_ev=obj.row['hybrid_error_ev'],iterations=len(hist),hessian_index_certified=False,
  nuclear_hybrid_hessian_available=False,script_sha256=digest(__file__))
 write(final,result);print('RESULT',result,flush=True);return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--molecule',choices=PROTOCOL['cases']);p.add_argument('--state',choices=PROTOCOL['states']);p.add_argument('--check-only',action='store_true');args=p.parse_args()
 OUT.mkdir(exist_ok=True);write(OUT/'protocol.json',PROTOCOL)
 for case in PROTOCOL['cases']:
  if args.molecule and args.molecule!=case:continue
  for state in PROTOCOL['states']:
   if args.state and args.state!=state:continue
   try:run(case,state,args.check_only)
   except Exception as exc:
    write(OUT/case/state/'failure.json',dict(host=HOST,error=str(exc),traceback=traceback.format_exc()))
    print('FAILURE',case,state,traceback.format_exc(),flush=True)
