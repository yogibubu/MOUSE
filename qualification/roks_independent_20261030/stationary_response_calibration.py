"""Both first and second DFT derivatives at the stationary DFT ground density.

Reuse all existing SCF references, embeddings and density-independent HF
response. Correct only the inconsistent density used by the XC response in
spectral transport. Two components retain the fixed 50/50 energy prescription.
No excited-state energy enters construction and no scalar offset is introduced.
"""
from remote_bootstrap import BASE,HOST
from pathlib import Path
import sys,json,time,numpy as np
from scipy.linalg import solve_sylvester
from threadpoolctl import threadpool_limits
from pyscf import dft,scf,ao2mo,lib
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import write,digest,key
from full_fast import features
ROOT=BASE/'results/stationary_response';ROOT.mkdir(exist_ok=True)

def calibrate(case,s,ctx):
 name=case['name'];out=ROOT/'calibration'/name;out.mkdir(parents=True,exist_ok=True)
 old=BASE/'results/full_response'/name;source=old/'model.npz';rfile=Path(case['reference'])/'reference.npz'
 request=key(dict(profile='stationary-reference-response-v1',script=digest(__file__),old_model=digest(source),reference=digest(rfile),lambda_exchange=.9))
 final=out/'model.npz'
 if final.exists():
  with np.load(final) as f:assert str(f['request_hash'])==request;return {k:f[k].copy() for k in f.files if k!='request_hash'},request
 start=time.perf_counter()
 with np.load(source) as f:operators=f['operators'];q0=f['q0']
 with np.load(rfile) as f:c=f['coefficients'];occ=f['occupations'];fd=f['fock']
 if (old/'embedding.npz').exists():
  with np.load(old/'embedding.npz') as f:lift=f['embedding']
 else:
  with np.load(Path('/home/enzo/MOUSE-roks-repair-20261030/results/spin_kernel')/name/'kernel.npz') as f:lift=f['embedding']
 mol=scf.chkfile.load_mol(str(Path(case['reference'])/'backend.chk'));mol.max_memory=40000
 dual=np.array([2*solve_sylvester(s.overlap,s.overlap,x) for x in operators]);n=len(s.overlap)
 linear_raw=np.einsum('aij,ij->a',dual,lift.T@fd@lift-s.fock)
 with np.load(BASE/'results/qualified_spectral_response'/name/'model.npz') as f:linear_spectral=f['linear']
 hf=out/'hf.npz'
 if hf.exists():
  with np.load(hf) as f:assert str(f['request_hash'])==request;charge=f['charge'];spin=f['spin']
 elif (old/'first_second_hf.npz').exists():
  with np.load(old/'first_second_hf.npz') as f:charge=f['charge'];spin=f['spin']
  np.savez_compressed(hf,request_hash=request,charge=charge,spin=spin)
 else:
  if (old/'projected_eri.npz').exists():
   with np.load(old/'projected_eri.npz') as f:eri=f['eri']
  elif (old/'partial_quadrature.npz').exists():
   with np.load(old/'partial_quadrature.npz') as f:eri=f['eri']
  else:eri=ao2mo.incore.general(mol.intor('int2e',aosym='s8'),(lift,lift,lift,lift),compact=False).reshape(n,n,n,n)
  d=dual.reshape(len(dual),-1);j=d@eri.reshape(n*n,n*n)@d.T;k=d@eri.transpose(0,2,1,3).reshape(n*n,n*n)@d.T
  spin=-.5*.225*k;charge=j+spin;np.savez_compressed(hf,request_hash=request,charge=charge,spin=spin)
 mf=dft.UKS(mol);mf.xc='PBE0';mf.verbose=0;mf.small_rho_cutoff=0;mf.max_memory=40000;mf.grids.level=3;mf.grids.build(with_non0tab=True)
 # The density differs from the previous embedded-native reference. Use the
 # exact cached DFT occupied density including its relaxed valence and core.
 dm0=(c*(.5*occ))@c.T
 progress=out/'quadrature_progress.npz';completed=0
 if progress.exists():
  with np.load(progress) as f:assert str(f['request_hash'])==request;charge=f['charge'];spin=f['spin'];completed=int(f['completed_blocks'])
 with threadpool_limits(limits=8,user_api='blas'):
  for i,(ao,mask,weight,coords) in enumerate(mf._numint.block_loop(mol,mf.grids,mol.nao_nr(),deriv=1,max_memory=256)):
   if i<completed:continue
   rho=mf._numint.eval_rho(mol,ao,dm0,mask,'GGA',hermi=1);fxc=mf._numint.eval_xc_eff('PBE0',np.array([rho,rho]),deriv=2,xctype='GGA')[2]
   fc=.25*(fxc[0,:,0]+fxc[0,:,1]+fxc[1,:,0]+fxc[1,:,1]);fs=.25*(fxc[0,:,0]-fxc[0,:,1]-fxc[1,:,0]+fxc[1,:,1]);phi=ao@lift
   for lo in range(0,len(weight),4096):
    hi=min(lo+4096,len(weight));feat=features(phi[:,lo:hi],dual);flat=feat.transpose(1,0,2).reshape(len(operators),-1)
    for target,fx in [(charge,fc),(spin,fs)]:
     weighted=(np.einsum('ijg,jag->aig',fx[:,:,lo:hi],feat,optimize=True)*weight[None,None,lo:hi]).reshape(len(operators),-1);target+=flat@weighted.T
   tmp=progress.with_suffix('.tmp.npz');np.savez_compressed(tmp,request_hash=request,charge=charge,spin=spin,completed_blocks=i+1);tmp.replace(progress)
   print('STATIONARY RESPONSE BLOCK',name,i+1,flush=True)
 native=np.array([np.einsum('aij,ij->a',dual,ctx.kernel(x)) for x in dual]).T;charge-=native
 assert max(np.max(abs(charge-charge.T)),np.max(abs(spin-spin.T)))<1e-6
 data=dict(operators=operators,q0=q0,charge=.5*(charge+charge.T),spin=.5*(spin+spin.T),linear_raw=linear_raw,linear_spectral=linear_spectral)
 np.savez_compressed(final,request_hash=request,**data)
 write(out/'calibration.json',dict(host=HOST,request_hash=request,source_sha256=digest(__file__),reference_sha256=digest(rfile),old_model_sha256=digest(source),seconds=time.perf_counter()-start,
  reference_density='actual stationary PBE0 RKS density, core and valence',first_and_second_derivatives_use_same_DFT_state=True,lambda_exchange=.9,gap_scale=1.,excited_energies_used=False,state_energy_offset=0.))
 return data,request
