"""Cached SCF and checkpointed Davidson for a fixed first-singlet reference root.

This changes only the numerical eigensolver, not the DFT/TDA operator. Save both
Ritz pairs and all completed trial-space operator products; reuse them on resume.
"""
from remote_bootstrap import HOST,BASE
from pathlib import Path
import sys,time,numpy as np
from pyscf import gto,dft,scf,lib
from threadpoolctl import threadpool_limits
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import write,key,digest

def reference(case,protocol_file):
 out=Path(case['reference']);out.mkdir(parents=True,exist_ok=True)
 request=key(dict(protocol=digest(protocol_file),geometry=digest(case['geometry']),model='PBE0/aug-cc-pVDZ/cart/grid3/checkpointed-first-singlet'))
 if (out/'request.json').exists():assert __import__('json').loads((out/'request.json').read_text())['request_hash']==request
 else:write(out/'request.json',dict(host=HOST,request_hash=request))
 if (out/'reference.npz').exists() and (out/'backend_tda.npz').exists():return
 text=Path(case['geometry']).read_text().splitlines();nat=int(text[0]);mol=gto.M(atom='\n'.join(text[2:2+nat]),basis='aug-cc-pvdz',unit='Angstrom',cart=True,verbose=0);mol.max_memory=40000
 mf=dft.RKS(mol);mf.xc='PBE0';mf.verbose=0;mf.max_memory=40000;mf.small_rho_cutoff=0;mf.grids.level=3
 mf.conv_tol=1e-10;mf.conv_tol_grad=3e-8;mf.max_cycle=120;mf.chkfile=str(out/'backend.chk');mf=scf.addons.remove_linear_dep_(mf,threshold=1e-7,lindep=1e-9)
 cached=out/'reference.npz'
 if cached.exists():
  with np.load(cached) as f:mf.mo_coeff=f['coefficients'];mf.mo_occ=f['occupations'];mf.mo_energy=f['orbital_energies_hartree'];mf.e_tot=float(f['energy_hartree'])
  mf.converged=True
 else:
  dm=None
  if (out/'backend.chk').exists():
   previous=scf.chkfile.load(str(out/'backend.chk'),'scf')
   if previous and 'mo_coeff' in previous:dm=(previous['mo_coeff']*previous['mo_occ'])@previous['mo_coeff'].T
  start=time.perf_counter();mf.kernel(dm0=dm)
  if not mf.converged:
   dm=mf.make_rdm1();mf=mf.newton();mf.conv_tol=1e-10;mf.conv_tol_grad=3e-8;mf.max_cycle=80;mf.kernel(dm0=dm)
  assert mf.converged
  s=mf.get_ovlp();c=mf.mo_coeff;fock=mf.get_fock(dm=mf.make_rdm1());grad=float(np.max(abs(mf.get_grad(c,mf.mo_occ,fock))));assert grad<1e-7
  assert np.max(abs(c.T@s@c-np.eye(c.shape[1])))<2e-7
  np.savez_compressed(cached,atomic_numbers=mol.atom_charges(),coordinates_bohr=mol.atom_coords(),overlap=s,coefficients=c,occupations=mf.mo_occ,orbital_energies_hartree=mf.mo_energy,fock=fock,energy_hartree=mf.e_tot)
  write(out/'scf.json',dict(host=HOST,request_hash=request,converged=True,orbital_residual=grad,energy_hartree=mf.e_tot,seconds=time.perf_counter()-start))
 if (out/'backend_tda.npz').exists():return
 td=mf.TDA();td.verbose=0;vind,diag=td.gen_vind();progress=out/'tda_progress.npz';cache_x=None;cache_ax=None;completed_before=0
 if progress.exists():
  with np.load(progress) as f:
   assert str(f['request_hash'])==request;x0=f['ritz_vectors'];cache_x=f['xs'];cache_ax=f['ax'];completed_before=int(f['iterations_total'])
 else:
  smallest=np.argsort(diag)[:4];x0=np.zeros((len(smallest)+1,len(diag)));x0[np.arange(len(smallest)),smallest]=1.
  x0[-1]=np.random.default_rng(20261030).normal(size=len(diag));x0[-1]/=np.linalg.norm(x0[-1])
 calls=[0]
 def hop(x):
  calls[0]+=1
  if cache_x is None:return vind(x[None])[0]
  proj=cache_x@x;remaining=x-proj@cache_x
  if np.linalg.norm(remaining)<1e-12:return proj@cache_ax
  return vind(remaining[None])[0]+proj@cache_ax
 def precondition(dx,e,x0):
  den=diag-e;den=np.where(abs(den)<1e-8,1e-8,den);return dx/den
 def checkpoint(env):
  tmp=progress.with_suffix('.tmp.npz');np.savez_compressed(tmp,request_hash=request,ritz_vectors=np.asarray(env['x0']),ritz_images=np.asarray(env['ax0']),energies=env['e'],residuals=env['dx_norm'],xs=np.asarray(env['xs']),ax=np.asarray(env['ax']),iterations_total=completed_before+env['icyc']+1);tmp.replace(progress)
 start=time.perf_counter()
 with threadpool_limits(limits=2,user_api='blas'):
  e,v=lib.davidson(hop,list(x0),precondition,tol=1e-16,lindep=1e-24,max_space=40,max_cycle=150,callback=checkpoint,verbose=0)
  residual=float(np.linalg.norm(vind(v[None])[0]-e*v));np.savez_compressed(out/'tda_final_pair.npz',energy=e,amplitude=v,residual=residual,request_hash=request)
  assert residual<1e-8 and e>0,(e,residual)
  np.savez_compressed(out/'backend_tda.npz',energies_hartree=np.array([e]),amplitudes=v[None]/np.linalg.norm(v))
 write(out/'tda_solver.json',dict(host=HOST,request_hash=request,reference_root=1,eigenvalue_hartree=float(e),residual=residual,lindep=1e-24,seconds=time.perf_counter()-start,operator_calls_current_run=calls[0],resumed=completed_before>0,SCF_repeated=False,source_sha256=digest(__file__)))
 print('REFERENCE COMPLETE',case['name'],float(e),residual,flush=True)
