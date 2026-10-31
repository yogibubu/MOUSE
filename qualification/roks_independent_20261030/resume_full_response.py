"""Resume completed scientific quadrature using equivalent blocked BLAS products.

The saved ERIs, first derivative and completed XC blocks are retained exactly.
The untouched original source defines the request hash and orbital solver.
"""
from remote_bootstrap import HOST,BASE
from pathlib import Path
import sys,time,numpy as np,json
from threadpoolctl import threadpool_limits
import full_response as full
from pyscf import dft,scf,lib
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import write,digest,key,plan


def features_blas(phi,dual):
 n=phi.shape[2];d=dual.reshape(len(dual),n*n);g=len(phi[0]);features=np.empty((4,len(dual),g))
 for k in range(4):
  pair=(phi[k,:,:,None]*phi[0,:,None,:]).reshape(g,n*n)
  features[k]=(d@pair.T)*(1. if k==0 else 2.)
 return features


def optimized(case,s,ctx):
 out=full.ROOT/case['name'];path=out/'model.npz'
 request=key(dict(reference=digest(Path(case['reference'])/'reference.npz'),script=digest(BASE/'full_response.py'),source_coordinates=s.coordinates_bohr.tolist()))
 if path.exists():
  with np.load(path) as f:assert str(f['request_hash'])==request;return {k:f[k].copy() for k in f.files},request
 start=time.perf_counter();checkpoint=out/'partial_quadrature.npz'
 with np.load(checkpoint) as f:
  assert str(f['request_hash'])==request
  arrays={k:f[k].copy() for k in f.files};completed=int(f['completed_blocks'])
 a=arrays['a'];dual=arrays['dual'];charge=arrays['charge'];spin=arrays['spin'];linear=arrays['linear'];p0=arrays['p0']
 b=.5*(s.overlap@a+a@s.overlap);error=float(np.max(abs(np.einsum('aij,bij->ab',b,dual)-np.eye(len(a)))))
 assert error<1e-8
 rng=np.random.default_rng(20261030);phi=rng.normal(size=(4,19,len(s.overlap)))
 old=np.array([np.einsum('gi,aij,gj->ag',p,dual,phi[0],optimize=True)*(1. if i==0 else 2.) for i,p in enumerate(phi)])
 new=features_blas(phi,dual);equivalence=float(np.max(abs(old-new)));assert equivalence<1e-10
 write(out/'blas_equivalence.json',dict(host=HOST,maximum_feature_difference=equivalence,passed=True,dual_error=error,checkpoint_sha256=digest(checkpoint)))
 folder=Path(case['reference'])
 with np.load(Path('/home/enzo/MOUSE-roks-repair-20261030/results/spin_kernel')/case['name']/'kernel.npz') as f:lift=f['embedding']
 with np.load(folder/'reference.npz') as f:c=f['coefficients'];occ=f['occupations']
 mol=scf.chkfile.load_mol(str(folder/'backend.chk'));mol.max_memory=40000
 mf=dft.UKS(mol);mf.xc='PBE0';mf.verbose=0;mf.max_memory=40000;mf.small_rho_cutoff=0;mf.grids.level=3;mf.grids.build(with_non0tab=True)
 ncore=int(np.sum(s.atomic_numbers-np.asarray(ctx.calculator._valence))/2);core=c[:,np.flatnonzero(occ==2)[:ncore]]
 dm0=core@core.T+.5*lift@p0@lift.T;blocks=0
 progress=out/'bulk_progress.npz'
 if progress.exists():
  with np.load(progress) as f:assert str(f['request_hash'])==request;charge=f['charge'];spin=f['spin'];completed=int(f['completed_blocks'])
 with threadpool_limits(limits=8,user_api='blas'):
  for i,(ao,mask,weight,coords) in enumerate(mf._numint.block_loop(mol,mf.grids,mol.nao_nr(),deriv=1,max_memory=2000)):
   if i<completed:blocks=i+1;continue
   if i==int(arrays['completed_blocks']):
    assert np.array_equal(coords,arrays['coords']) and np.array_equal(weight,arrays['weight']),'quadrature partition differs from saved checkpoint'
   rho=mf._numint.eval_rho(mol,ao,dm0,mask,'GGA',hermi=1)
   fxc=mf._numint.eval_xc_eff('PBE0',np.array([rho,rho]),deriv=2,xctype='GGA')[2]
   fc=.25*(fxc[0,:,0]+fxc[0,:,1]+fxc[1,:,0]+fxc[1,:,1]);fs=.25*(fxc[0,:,0]-fxc[0,:,1]-fxc[1,:,0]+fxc[1,:,1])
   phi=ao@lift
   for lo in range(0,len(weight),4096):
    hi=min(lo+4096,len(weight));features=features_blas(phi[:,lo:hi],dual);flat=features.transpose(1,0,2).reshape(len(a),-1)
    for target,fx in [(charge,fc),(spin,fs)]:
     weighted=(np.einsum('ijg,jag->aig',fx[:,:,lo:hi],features,optimize=True)*weight[None,None,lo:hi]).reshape(len(a),-1)
     target+=flat@weighted.T
    print('BLAS QUADRATURE',case['name'],i,hi,len(weight),flush=True)
   blocks=i+1;temporary=progress.with_suffix('.tmp.npz');np.savez_compressed(temporary,request_hash=request,charge=charge,spin=spin,completed_blocks=blocks);temporary.replace(progress)
 native=np.array([np.einsum('aij,ij->a',dual,ctx.kernel(x)) for x in dual]).T;charge-=native
 assert max(np.max(abs(charge-charge.T)),np.max(abs(spin-spin.T)))<1e-6
 data=dict(operators=a,charge=.5*(charge+charge.T),spin=.5*(spin+spin.T),linear=linear,q0=np.einsum('aij,ij->a',b,p0))
 np.savez_compressed(path,request_hash=request,**data)
 write(out/'calibration.json',dict(host=HOST,request_hash=request,feature_count=len(a),seconds_resume=time.perf_counter()-start,
  preserved_quadrature_blocks=int(arrays['completed_blocks']),total_blocks=blocks,eri_recomputed=False,first_derivative_recomputed=False,
  scalar_energy_offset=0.,reference_density='DFT frozen core plus embedded native GFN2 ground valence density',
  excited_reference_energies_used=False,scope='vertical-only complete AO-pair response diagnostic',
  original_source_sha256=digest(BASE/'full_response.py'),resume_source_sha256=digest(__file__)))
 return data,request

full.calibrate=optimized
if __name__=='__main__':
 case=next(c for c in plan() if c['name']=='uracil')
 with threadpool_limits(limits=2,user_api='blas'):full.run(case)
