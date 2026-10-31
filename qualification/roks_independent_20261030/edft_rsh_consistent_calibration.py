"""Consistent RSH first and second derivative calibration; no fitted state energies."""
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

RSH_XC='RSH(1;-.625;.2)+.625*GGA_X_PBE+GGA_C_PBE'
ROOT=BASE/'results/edft_rsh_consistent';ROOT.mkdir(exist_ok=True)

def _calibrate(case,s,ctx):
 out=full.ROOT/case['name'];out.mkdir(exist_ok=True);final=out/'model.npz'
 request=key(dict(profile='consistent-rsh-ao-pair-v1',reference=digest(Path(case['reference'])/'reference.npz'),source=digest(__file__),
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
 mf=dft.UKS(mol);mf.xc=RSH_XC;mf.verbose=0;mf.max_memory=40000;mf.small_rho_cutoff=0;mf.grids.level=3;mf.grids.build(with_non0tab=True)
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
  with mol.with_range_coulomb(.2):
   lr=ao2mo.incore.general(mol.intor('int2e',aosym='s8'),(lift,lift,lift,lift),compact=False).reshape(n,n,n,n)
  np.savez_compressed(out/'long_range_eri.npz',request_hash=request,eri=lr)
  klr=d@lr.transpose(0,2,1,3).reshape(n*n,n*n)@d.T
  spin=-.5*.9*(.375*k+.625*klr);charge=j+spin
  np.savez_compressed(prepared,request_hash=request,linear=linear,spin=spin,charge=charge)
 progress=out/'quadrature_progress.npz';completed=0
 if progress.exists():
  with np.load(progress) as f:assert str(f['request_hash'])==request;spin=f['spin'];charge=f['charge'];completed=int(f['completed_blocks'])
 with threadpool_limits(limits=8,user_api='blas'):
  for i,(ao,mask,weight,coords) in enumerate(mf._numint.block_loop(mol,mf.grids,mol.nao_nr(),deriv=1,max_memory=256)):
   if i<completed:continue
   rho=mf._numint.eval_rho(mol,ao,dm0,mask,'GGA',hermi=1);fxc=mf._numint.eval_xc_eff(RSH_XC,np.array([rho,rho]),deriv=2,xctype='GGA')[2]
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
  energy_offset=0.,excited_reference_energies_used=False,profile='consistent-rsh-ao-pair-v1',source_sha256=digest(__file__)))
 return data,request

def reference(case):
 folder=ROOT/'references'/case['name'];folder.mkdir(parents=True,exist_ok=True)
 source=Path(case['reference'])/'reference.npz';request=key(dict(xc=RSH_XC,source=digest(source),grid=3))
 final=folder/'reference.npz'
 if final.exists():
  record=json.loads((folder/'reference.json').read_text());assert record['request_hash']==request
  return dict(case,reference=str(folder))
 mol=scf.chkfile.load_mol(str(Path(case['reference'])/'backend.chk'));mol.max_memory=40000
 mf=dft.RKS(mol);mf.xc=RSH_XC;mf.grids.level=3;mf.small_rho_cutoff=0;mf.verbose=4
 mf.max_memory=40000;mf.conv_tol=1e-11;mf.conv_tol_grad=1e-8;mf.max_cycle=150;mf.chkfile=str(folder/'backend.chk')
 assert np.allclose(mf._numint.rsh_and_hybrid_coeff(RSH_XC,mol.spin),[.2,1.,.375])
 if (folder/'backend.chk').exists():dm=mf.init_guess_by_chkfile(str(folder/'backend.chk'))
 else:
  with np.load(source) as f:dm=(f['coefficients']*f['occupations'])@f['coefficients'].T
 start=time.perf_counter();energy=mf.kernel(dm0=dm)
 assert mf.converged,'RSH reference failed; SCF checkpoint retained'
 density=mf.make_rdm1();fock=mf.get_fock(dm=density)
 np.savez_compressed(final,atomic_numbers=mol.atom_charges(),coordinates_bohr=mol.atom_coords(),overlap=mf.get_ovlp(),coefficients=mf.mo_coeff,
                     occupations=mf.mo_occ,orbital_energies_hartree=mf.mo_energy,fock=fock,energy_hartree=energy)
 write(folder/'reference.json',dict(host=HOST,request_hash=request,xc=RSH_XC,grid=3,seconds=time.perf_counter()-start,converged=True,source_sha256=digest(source)))
 return dict(case,reference=str(folder))

def calibrate(case,s,ctx):
 case=reference(case);full.ROOT=ROOT/'calibration';full.ROOT.mkdir(exist_ok=True)
 data,raw_request=_calibrate(case,s,ctx);out=full.ROOT/case['name'];data['linear_raw']=data['linear'].copy()
 with np.load(out/'embedding.npz') as f:lift=f['embedding']
 with np.load(Path(case['reference'])/'reference.npz') as f:cd=f['coefficients'];sd=f['overlap'];od=f['occupations'];ed=f['orbital_energies_hartree']
 from scipy.optimize import linear_sum_assignment
 cross=s.coefficients.T@lift.T@sd@cd;occupied=np.flatnonzero(s.alpha_occupations==1);nc=len(np.flatnonzero(od==2))-len(occupied)
 selected=np.zeros(len(s.overlap),int);score=np.zeros(len(selected));margin=score.copy()
 for sector in [1,0]:
  ids=np.flatnonzero(s.alpha_occupations==sector);refs=np.flatnonzero(od==2)[nc:] if sector else np.flatnonzero(od==0)
  matrix=cross[np.ix_(ids,refs)];_,j=linear_sum_assignment(-matrix*matrix)
  for i,k in enumerate(ids):selected[k]=refs[j[i]];score[k]=abs(matrix[i,j[i]]);margin[k]=score[k]-np.max(np.delete(abs(matrix[i]),j[i]),initial=0.)
 good=(score>=.7)&(margin>=.05);ref_origin=float(np.mean(ed[selected[occupied]]));src_origin=float(np.mean(s.orbital_energies_hartree[occupied]))
 delta=np.zeros(len(selected));delta[good]=(ed[selected[good]]-ref_origin)-(s.orbital_energies_hartree[good]-src_origin)
 levels=s.orbital_energies_hartree+delta
 assert np.max(levels[occupied])<np.min(levels[s.alpha_occupations==0]),'RSH spectral gap lost'
 vao=(s.overlap@s.coefficients*delta)@s.coefficients.T@s.overlap
 dual=np.array([2*solve_sylvester(s.overlap,s.overlap,a) for a in data['operators']]);data['linear_spectral']=np.einsum('aij,ij->a',dual,vao)
 data['qualified_mapping']=good;request=key(dict(raw=raw_request,reference=digest(Path(case['reference'])/'reference.npz'),mapping=digest(__file__)))
 write(out/'mapping.json',dict(host=HOST,request_hash=request,reference_indices=selected,overlaps=score,margins=margin,qualified=good,
                             delta_hartree=delta,state_energy_offset=0.,reference_functional=RSH_XC))
 return data,request
