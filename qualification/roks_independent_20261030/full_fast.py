"""Generic full AO-pair calibration, checkpointed, no fitted state energies."""
from remote_bootstrap import HOST,BASE
from pathlib import Path
from types import SimpleNamespace
import sys,json,time,numpy as np
from scipy.linalg import eigh,solve_sylvester
from threadpoolctl import threadpool_limits
from pyscf import dft,scf,gto,ao2mo,lib
import full_response as full
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import write,digest,key
from extent_support import source_molecule
lib.num_threads(8)

def features(phi,dual):
 n=phi.shape[2];g=len(phi[0]);d=dual.reshape(len(dual),n*n);result=np.empty((4,len(dual),g))
 for k in range(4):
  pair=(phi[k,:,:,None]*phi[0,:,None,:]).reshape(g,n*n)
  result[k]=(d@pair.T)*(1. if k==0 else 2.)
 return result

def invsqrt(a):
 v,u=eigh(a)
 if np.min(v)<1e-9:raise ValueError('singular embedding metric')
 return (u*v**-.5)@u.T

def calibrate(case,s,ctx):
 out=full.ROOT/case['name'];out.mkdir(exist_ok=True);final=out/'model.npz'
 request=key(dict(profile='full-ao-pair-v1',reference=digest(Path(case['reference'])/'reference.npz'),source=digest(__file__),
  lambda_exchange=.9,coordinates=s.coordinates_bohr.tolist()))
 if final.exists():
  with np.load(final) as f:assert str(f['request_hash'])==request;return {k:f[k].copy() for k in f.files},request
 start=time.perf_counter();folder=Path(case['reference']);mol=scf.chkfile.load_mol(str(folder/'backend.chk'));mol.max_memory=40000
 with np.load(folder/'reference.npz') as f:c=f['coefficients'];occ=f['occupations'];sd=f['overlap']
 n=len(s.overlap);ncore=int(np.sum(s.atomic_numbers-np.asarray(ctx.calculator._valence))/2);core=c[:,np.flatnonzero(occ==2)[:ncore]]
 embedding=out/'embedding.npz'
 if embedding.exists():
  with np.load(embedding) as f:assert str(f['request_hash'])==request;lift=f['embedding']
 else:
  small,t=source_molecule(s);cross=gto.intor_cross('int1e_ovlp',mol,small)@t
  x=np.linalg.solve(sd,cross)-core@(core.T@cross);v,u=eigh(s.overlap);sqrt=(u*np.sqrt(v))@u.T
  lift=x@invsqrt(x.T@sd@x)@sqrt;error=float(np.max(abs(lift.T@sd@lift-s.overlap)));assert error<1e-8
  np.savez_compressed(embedding,request_hash=request,embedding=lift,isometry_error=error)
 a=full.symmetric_basis(n);dual=np.array([2*solve_sylvester(s.overlap,s.overlap,x) for x in a]);b=.5*(s.overlap@a+a@s.overlap)
 assert np.max(abs(np.einsum('aij,bij->ab',b,dual)-np.eye(len(a))))<1e-8
 p0=(s.coefficients*(s.alpha_occupations+s.beta_occupations))@s.coefficients.T
 mf=dft.UKS(mol);mf.xc='PBE0';mf.verbose=0;mf.max_memory=40000;mf.small_rho_cutoff=0;mf.grids.level=3;mf.grids.build(with_non0tab=True)
 dm0=core@core.T+.5*lift@p0@lift.T
 prepared=out/'first_second_hf.npz'
 if prepared.exists():
  with np.load(prepared) as f:assert str(f['request_hash'])==request;linear=f['linear'];spin=f['spin'];charge=f['charge']
 else:
  mf._eri=mol.intor('int2e',aosym='s8')
  fock=mf.get_hcore()+mf.get_veff(mol,np.array([dm0,dm0]))[0]
  linear=np.einsum('aij,ij->a',dual,lift.T@fock@lift-s.fock)
  eri=ao2mo.incore.general(mf._eri,(lift,lift,lift,lift),compact=False).reshape(n,n,n,n)
  np.savez_compressed(out/'projected_eri.npz',request_hash=request,eri=eri)
  d=dual.reshape(len(a),-1);j=d@eri.reshape(n*n,n*n)@d.T;k=d@eri.transpose(0,2,1,3).reshape(n*n,n*n)@d.T
  spin=-.5*.225*k;charge=j+spin
  np.savez_compressed(prepared,request_hash=request,linear=linear,spin=spin,charge=charge)
 progress=out/'quadrature_progress.npz';completed=0
 if progress.exists():
  with np.load(progress) as f:assert str(f['request_hash'])==request;spin=f['spin'];charge=f['charge'];completed=int(f['completed_blocks'])
 with threadpool_limits(limits=8,user_api='blas'):
  for i,(ao,mask,weight,coords) in enumerate(mf._numint.block_loop(mol,mf.grids,mol.nao_nr(),deriv=1,max_memory=256)):
   if i<completed:continue
   rho=mf._numint.eval_rho(mol,ao,dm0,mask,'GGA',hermi=1);fxc=mf._numint.eval_xc_eff('PBE0',np.array([rho,rho]),deriv=2,xctype='GGA')[2]
   fc=.25*(fxc[0,:,0]+fxc[0,:,1]+fxc[1,:,0]+fxc[1,:,1]);fs=.25*(fxc[0,:,0]-fxc[0,:,1]-fxc[1,:,0]+fxc[1,:,1]);phi=ao@lift
   for lo in range(0,len(weight),4096):
    hi=min(lo+4096,len(weight));feat=features(phi[:,lo:hi],dual);flat=feat.transpose(1,0,2).reshape(len(a),-1)
    for target,fx in [(charge,fc),(spin,fs)]:
     weighted=(np.einsum('ijg,jag->aig',fx[:,:,lo:hi],feat,optimize=True)*weight[None,None,lo:hi]).reshape(len(a),-1);target+=flat@weighted.T
   temp=progress.with_suffix('.tmp.npz');np.savez_compressed(temp,request_hash=request,charge=charge,spin=spin,completed_blocks=i+1);temp.replace(progress)
   print('FULL BLOCK',case['name'],i+1,flush=True)
 native=np.array([np.einsum('aij,ij->a',dual,ctx.kernel(x)) for x in dual]).T;charge-=native
 assert max(np.max(abs(charge-charge.T)),np.max(abs(spin-spin.T)))<1e-6
 data=dict(operators=a,charge=.5*(charge+charge.T),spin=.5*(spin+spin.T),linear=linear,q0=np.einsum('aij,ij->a',b,p0))
 temp=final.with_suffix('.tmp.npz');np.savez_compressed(temp,request_hash=request,**data);temp.replace(final)
 write(out/'calibration.json',dict(host=HOST,request_hash=request,feature_count=len(a),seconds=time.perf_counter()-start,
  energy_offset=0.,excited_reference_energies_used=False,profile='full-ao-pair-v1',source_sha256=digest(__file__)))
 return data,request
