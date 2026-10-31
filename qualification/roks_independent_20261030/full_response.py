"""Vertical-only diagnostic: complete projected DFT Fock and charge/spin Hessians.

This deliberately has no nuclear-derivative interface. No state energy offset
is used. Full AO-pair moments retain nonlocal response lost by shell projection.
"""
from remote_bootstrap import HOST,BASE
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
import sys,json,numpy as np,time,argparse,traceback,hashlib
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import plan,state_from,write,digest,key,EV
from matrix_gfn2 import GFN2Calculator,FrozenResponseReference,evaluate_gfn2_roks_singlet
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.deltascf_derivatives import save_derivative_point,load_derivative_point
from matrix_apoc.deltascf import singlet_csf_overlap
from scipy.linalg import solve_sylvester
from pyscf import dft,scf,ao2mo,lib
lib.num_threads(8);ROOT=BASE/'results/full_response';ROOT.mkdir(exist_ok=True)

def symmetric_basis(n):
 rows,cols=np.triu_indices(n);a=np.zeros((len(rows),n,n))
 for k,(i,j) in enumerate(zip(rows,cols)):
  a[k,i,j]=a[k,j,i]=1. if i==j else 1/np.sqrt(2)
 return a

class MetricTerm:
 def __init__(self,s,a,k,q0=None,v=None):
  self.b=.5*(s@a+a@s);self.k=k;self.q0=np.zeros(len(a)) if q0 is None else q0
  self.v=np.zeros(len(a)) if v is None else v
 def evaluate(self,p):
  q=np.einsum('aij,ij->a',self.b,p);dq=q-self.q0;field=self.k@dq+self.v
  return np.einsum('a,aij->ij',field,self.b),.5*float(dq@self.k@dq)+float(self.v@q)
 def response(self,p):return np.einsum('a,aij->ij',self.k@np.einsum('aij,ij->a',self.b,p),self.b)

class FullSpin:
 vertical_only=True
 def __init__(self,base,term):self.base=base;self.term=term
 def evaluate(self,p):
  f,e=self.term.evaluate(p[0]-p[1]);return np.array([f,-f]),e,np.zeros(len(self.base.coupling_hartree))

class VerticalReference(FrozenResponseReference):
 def __init__(self,s,partition,data,request):
  ns=len(partition.shell_atom)
  super().__init__(s.atomic_numbers,partition.orbital_shell,np.zeros((ns,ns)),request,.9,np.zeros((ns,ns)),np.zeros(ns))
  object.__setattr__(self,'data',data);object.__setattr__(self,'anchor_coordinates',s.coordinates_bohr.copy())
 @property
 def fingerprint(self):return hashlib.sha256(super().fingerprint.encode()+b'full-vertical').hexdigest()
 def bind(self,state,partition):
  if np.max(abs(state.coordinates_bohr-self.anchor_coordinates))>1e-10:raise ValueError('vertical-only diagnostic; no displaced geometries')
  base=super().bind(state,partition)
  return FullSpin(base,MetricTerm(state.overlap,self.data['operators'],self.data['spin']))
 def bind_charge(self,state,partition):return MetricTerm(state.overlap,self.data['operators'],self.data['charge'],self.data['q0'],self.data['linear'])

def calibrate(case,s,ctx):
 out=ROOT/case['name'];out.mkdir(exist_ok=True);path=out/'model.npz'
 request=key(dict(reference=digest(Path(case['reference'])/'reference.npz'),script=digest(__file__),source_coordinates=s.coordinates_bohr.tolist()))
 if path.exists():
  with np.load(path) as f:assert str(f['request_hash'])==request;return {k:f[k].copy() for k in f.files},request
 start=time.perf_counter();folder=Path(case['reference']);old=Path('/home/enzo/MOUSE-roks-repair-20261030/results/spin_kernel')/case['name']/'kernel.npz'
 with np.load(old) as f:lift=f['embedding'];assert str(f['reference_sha256'])==digest(folder/'reference.npz')
 with np.load(folder/'reference.npz') as f:c=f['coefficients'];occ=f['occupations']
 mol=scf.chkfile.load_mol(str(folder/'backend.chk'));mol.max_memory=40000
 mf=dft.UKS(mol);mf.xc='PBE0';mf.verbose=0;mf.max_memory=40000;mf.small_rho_cutoff=0;mf.grids.level=3;mf.grids.build(with_non0tab=True)
 mf._eri=mol.intor('int2e',aosym='s8');n=len(s.overlap);a=symmetric_basis(n)
 dual=np.array([2*solve_sylvester(s.overlap,s.overlap,x) for x in a]);b=.5*(s.overlap@a+a@s.overlap)
 assert np.max(abs(np.einsum('aij,bij->ab',b,dual)-np.eye(len(a))))<1e-8
 p0=(s.coefficients*(s.alpha_occupations+s.beta_occupations))@s.coefficients.T
 ncore=int(np.sum(s.atomic_numbers-np.asarray(ctx.calculator._valence))/2);core=c[:,np.flatnonzero(occ==2)[:ncore]]
 dm0=core@core.T+.5*lift@p0@lift.T
 fock=mf.get_hcore()+mf.get_veff(mol,np.array([dm0,dm0]))[0]
 linear=np.einsum('aij,ij->a',dual,lift.T@fock@lift-s.fock)
 print('FULL ERI',case['name'],n,len(a),flush=True)
 eri=ao2mo.incore.general(mf._eri,(lift,lift,lift,lift),compact=False).reshape(n,n,n,n)
 d=dual.reshape(len(a),-1);j=d@eri.reshape(n*n,n*n)@d.T;k=d@eri.transpose(0,2,1,3).reshape(n*n,n*n)@d.T
 spin=-.5*.225*k;charge=j+spin
 blocks=0
 for ao,mask,weight,coords in mf._numint.block_loop(mol,mf.grids,mol.nao_nr(),deriv=1,max_memory=2000):
  rho=mf._numint.eval_rho(mol,ao,dm0,mask,'GGA',hermi=1)
  fxc=mf._numint.eval_xc_eff('PBE0',np.array([rho,rho]),deriv=2,xctype='GGA')[2]
  fc=.25*(fxc[0,:,0]+fxc[0,:,1]+fxc[1,:,0]+fxc[1,:,1])
  fs=.25*(fxc[0,:,0]-fxc[0,:,1]-fxc[1,:,0]+fxc[1,:,1])
  phi=ao@lift;features=np.empty((4,len(a),len(weight)))
  features[0]=np.einsum('gi,aij,gj->ag',phi[0],dual,phi[0],optimize=True)
  for xyz in range(1,4):features[xyz]=2*np.einsum('gi,aij,gj->ag',phi[xyz],dual,phi[0],optimize=True)
  flat=features.transpose(1,0,2).reshape(len(a),-1)
  for target,fx in [(charge,fc),(spin,fs)]:
   weighted=np.einsum('ijg,jag,g->aig',fx,features,weight,optimize=True).reshape(len(a),-1)
   target+=flat@weighted.T
  blocks+=1
  if blocks%5==0:print('FULL GRID',case['name'],blocks,flush=True)
 # Remove the native charge Hessian already present in GFN2.
 native=np.array([np.einsum('aij,ij->a',dual,ctx.kernel(x)) for x in dual]).T
 charge-=native
 assert max(np.max(abs(charge-charge.T)),np.max(abs(spin-spin.T)))<1e-6
 data=dict(operators=a,charge=.5*(charge+charge.T),spin=.5*(spin+spin.T),linear=linear,q0=np.einsum('aij,ij->a',b,p0))
 np.savez_compressed(path,request_hash=request,**data)
 write(out/'calibration.json',dict(host=HOST,request_hash=request,feature_count=len(a),seconds=time.perf_counter()-start,scalar_energy_offset=0.,
  reference_density='DFT frozen core plus embedded native GFN2 ground valence density',excited_reference_energies_used=False,scope='vertical-only complete AO-pair response diagnostic'))
 return data,request

def run(case):
 r0,xyz=state_from(case['targets'][0]);s=r0.reference_state
 calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
 ground=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=r0.ground_energy_hartree);ctx=prepare_gfn2_deltascf(calc,xyz,ground_state=ground)
 data,request=calibrate(case,s,ctx);ref=VerticalReference(s,ctx.kernel.partition,data,request)
 ctx=prepare_gfn2_deltascf(calc,xyz,ground_state=ground,spin_reference=ref)
 rows=[]
 for target in case['targets']:
  out=ROOT/case['name']/target['id'];out.mkdir(exist_ok=True)
  if (out/'result.json').exists():rows.append(json.loads((out/'result.json').read_text()));continue
  old,_=state_from(target);good=False;attempts=[]
  for index in list(dict.fromkeys([old.orbital_saddle_index,0,1,2,3])):
   dest=out/f'index_{index}';dest.mkdir(exist_ok=True);req=key(dict(reference=request,target=target,index=index))
   try:
    cp=dest/'converged.npz';latest=dest/'latest.npz';start=time.perf_counter()
    if cp.exists():r=load_derivative_point(cp,request_hash=req,coordinates_angstrom=xyz,require_gradient=False)
    else:
     c=s.coefficients.copy()
     if latest.exists():
      with np.load(latest) as f:assert str(f['request_hash'])==req;c=f['coefficients']
     fm,em=ctx.fock_and_energy(np.array([(c*o)@c.T for o in old.occupations]));ft,et=ctx.fock_and_energy(np.array([(c*o)@c.T for o in old.triplet_occupations]))
     seed=replace(old,common_coefficients=c,fock_mixed=fm,fock_triplet=ft,energy_hartree=2*em-et,converged=False,orbital_saddle_index=None,target_orbital_saddle_index=None,spin_reference_fingerprint=ref.fingerprint,gradient_hartree_per_bohr=None)
     def checkpoint(point):
      np.savez_compressed(dest/'latest.tmp.npz',request_hash=req,coefficients=point.point);(dest/'latest.tmp.npz').replace(latest)
      write(dest/'history.json',list(point.history))
     r=evaluate_gfn2_roks_singlet(calc,xyz,*old.occupations,ground_state=ground,reference_result=seed,spin_reference=ref,minimum_state_overlap=0.,
      orbital_solver='gmf',saddle_index=index,orbital_tolerance=1e-9,maximum_iterations=100,maximum_step=.2,freeze_and_release=True,freeze_iterations=8,checkpoint=checkpoint)
     save_derivative_point(cp,r,request_hash=req,coordinates_angstrom=xyz)
    score=singlet_csf_overlap(replace(r,common_coefficients=s.coefficients).to_apoc(),r.to_apoc(),s.overlap)[0]
    good=bool(score>=.7 and r.orbital_residual<=1e-9 and r.orbital_saddle_index==index)
    row=dict(host=HOST,molecule=case['name'],state=target['id'],qualified=good,excitation_ev=r.excitation_energy_hartree*EV,tbe_ev=target['tbe_ev'],error_ev=r.excitation_energy_hartree*EV-target['tbe_ev'],
     csf_overlap=score,orbital_residual=r.orbital_residual,index=index,seconds_current_run=time.perf_counter()-start,scalar_energy_offset=0.,scope='vertical only; no nuclear derivatives',checkpoint=str(cp),request_hash=req)
    attempts.append(row);write(dest/'result.json',row);print('FULL RESPONSE',row,flush=True)
    if good:break
   except Exception as exc:write(dest/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()));print(traceback.format_exc(),flush=True)
  if not good:row=dict(molecule=case['name'],state=target['id'],qualified=False,attempts=attempts)
  write(out/'result.json',row);rows.append(row)
 write(ROOT/case['name']/'result.json',dict(host=HOST,records=rows))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--molecule',default='nitroxyl');args=p.parse_args()
 for c in plan():
  if c['name']==args.molecule:run(c)
