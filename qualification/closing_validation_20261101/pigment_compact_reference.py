"""Checkpointed frozen-core CAM anchor in the native pigment valence image.

DF Coulomb factors and quadrature are prepared once. Iterations contract only
native-sized matrices; no new full-space DFT iteration or TDA is invoked.
"""
import socket,subprocess,argparse,time,json,os
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
assert subprocess.check_output(['scutil','--get','ComputerName'],text=True).strip()=='Mac Studio di Vincenzo'
from prepare_fresh import HERE
from pigment_reference_embedding import write,sha,power
from matrix_gfn2.electronic_state import GFN2OrbitalState,GFN2OrbitalBasis
from dataclasses import fields
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from pyscf import scf,dft
from threadpoolctl import threadpool_limits

def load_state(name):
 folder=HERE/'pigment_preflight'/name
 path=folder/'stationary_native.npz' if (folder/'stationary_native.npz').exists() else folder/'native_ground.npz'
 with np.load(path) as z:
  basis=GFN2OrbitalBasis(**{f.name:z['basis_'+f.name].copy() for f in fields(GFN2OrbitalBasis)})
  kw={f.name:z[f.name].copy() for f in fields(GFN2OrbitalState) if f.name!='basis'}
 kw['model_id']=str(kw['model_id']);kw['ao_shell']=tuple(kw['ao_shell'].tolist())
 for field in ['electron_count','charge','multiplicity']:kw[field]=int(kw[field])
 kw['total_energy_hartree']=float(kw['total_energy_hartree']);return GFN2OrbitalState(**kw,basis=basis)

class Compact:
 def __init__(self,name):
  self.name=name;self.out=HERE/'pigment_compact_reference'/name;self.out.mkdir(parents=True,exist_ok=True);self.state=load_state(name);self.no=self.state.electron_count//2;self.n=len(self.state.overlap)
  root=HERE/'pigment_dft_reference'/name
  with np.load(HERE/'pigment_embedding'/name/'embedding.npz') as z:self.lift=z['embedding'].copy()
  certificate=json.loads((HERE/'pigment_embedding'/name/'certificate.json').read_text());self.nc=certificate['frozen_core_orbitals']
  with np.load(root/'PBE0/reference.npz') as z:self.core=z['coefficients'][:,:self.nc].copy()
  self.mol=scf.chkfile.load_mol(str(root/'PBE0/backend.chk'));self.mf=dft.RKS(self.mol);self.mf.xc='CAM-B3LYP';self.mf.verbose=0;self.mf.small_rho_cutoff=0.;self.mf.grids.level=3;self.mf.grids.build(with_non0tab=True)
  integral=HERE/'pigment_integrals'/name;assert (integral/'long_range_complete.json').exists()
  self.full=np.load(integral/'coulomb_factors.npy',mmap_mode='r');self.long=np.load(integral/'long_range_factors.npy',mmap_mode='r');self.a=self.full[:,self.nc:,self.nc:];self.al=self.long[:,self.nc:,self.nc:]
  path=self.out/'hcore.npy'
  if path.exists():self.hcore=np.load(path)
  else:
   self.hcore=self.lift.T@(self.mol.intor('int1e_kin')+self.mol.intor('int1e_nuc'))@self.lift
   for lo in range(0,len(self.full),64):
    block=self.full[lo:lo+64];a=block[:,self.nc:,self.nc:];core_charge=2*np.trace(block[:,:self.nc,:self.nc],axis1=1,axis2=2);self.hcore+=np.einsum('L,Lij->ij',core_charge,a)
    mixed=block[:,self.nc:,:self.nc];self.hcore-=.19*np.einsum('Lia,Lja->ij',mixed,mixed,optimize=True)
   for lo in range(0,len(self.long),64):
    mixed=self.long[lo:lo+64,self.nc:,:self.nc];self.hcore-=.46*np.einsum('Lia,Lja->ij',mixed,mixed,optimize=True)
   np.save(path,self.hcore)
  self.grid=[];grid=self.out/'grid';grid.mkdir(exist_ok=True)
  for k,(ao,mask,weight,coordinates) in enumerate(self.mf._numint.block_loop(self.mol,self.mf.grids,self.mol.nao_nr(),deriv=1,max_memory=256)):
   path=grid/f'chunk_{k:04d}.npz'
   if not path.exists():
    phi=ao@self.lift;pc=ao@self.core;core=np.empty((4,len(weight)));core[0]=2*np.einsum('gi,gi->g',pc[0],pc[0])
    for axis in range(1,4):core[axis]=4*np.einsum('gi,gi->g',pc[axis],pc[0])
    temporary=path.with_suffix('.tmp.npz');np.savez(temporary,phi=phi,core=core,weight=weight,coordinates=coordinates);temporary.replace(path)
   self.grid.append(path)
  write(self.out/'grid_complete.json',dict(chunks=len(self.grid),points=len(self.mf.grids.weights),basis='native valence image',grid_level=3,core_orbitals=self.nc))
  self.cache={}
 @staticmethod
 def rho(phi,p):
  a=phi[0]@p;rho=np.empty((4,phi.shape[1]));rho[0]=np.einsum('gi,gi->g',a,phi[0])
  for k in range(1,4):rho[k]=2*np.einsum('gi,gi->g',a,phi[k])
  return rho
 def xc(self,p):
  energy=0.;v=np.zeros_like(p)
  for path in self.grid:
   with np.load(path) as z:phi=z['phi'];weight=z['weight'];rho=z['core']+self.rho(phi,p)
   exc,vxc,*_=self.mf._numint.eval_xc_eff('CAM-B3LYP',rho,deriv=1,xctype='GGA');energy+=float(weight@(rho[0]*exc));v+=phi[0].T@((weight*vxc[0])[:,None]*phi[0])
   for k in range(1,4):
    part=phi[k].T@((weight*vxc[k])[:,None]*phi[0]);v+=part+part.T
  return energy,.5*(v+v.T)
 @staticmethod
 def jk(factors,p,coulomb=True):
  j=np.zeros_like(p);k=np.zeros_like(p)
  for lo in range(0,len(factors),64):
   b=np.asarray(factors[lo:lo+64]);t=b@p;k+=np.einsum('Lij,Lkj->ik',t,b,optimize=True)
   if coulomb:j+=np.einsum('L,Lij->ij',np.einsum('Lij,ij->L',b,p),b)
  return j,k
 def evaluate(self,p):
  if 'p' not in self.cache or not np.array_equal(p,self.cache['p']):
   start=time.monotonic();j,k=self.jk(self.a,p);_,kl=self.jk(self.al,p,False);exc,vxc=self.xc(p);exchange=.19*k+.46*kl;f=self.hcore+j-.5*exchange+vxc;e=float(np.sum(self.hcore*p)+.5*np.sum(p*j)-.25*np.sum(p*exchange)+exc);self.cache=dict(p=p.copy(),fock=f,energy=e)
   print(self.name,'compact Fock',time.monotonic()-start,flush=True)
  return self.cache['fock'],self.cache['energy']

def run(name):
 out=HERE/'pigment_compact_reference'/name;out.mkdir(parents=True,exist_ok=True)
 if (out/'certificate.json').exists():return
 start=time.monotonic();a=Compact(name);s=a.state
 mf=scf.RHF(a.mol);mf.verbose=0;mf.chkfile=None;mf.get_hcore=lambda *x,**k:a.hcore;mf.get_ovlp=lambda *x,**k:s.overlap
 mf.get_veff=lambda mol=None,dm=None,*x,**k:a.evaluate(dm)[0]-a.hcore
 mf.energy_elec=lambda dm=None,h1e=None,vhf=None:(a.evaluate(dm)[1],0.);mf.energy_nuc=lambda *x,**k:0.
 def occupations(e,c=None):
  o=np.zeros(len(e));o[np.argsort(e)[:a.no]]=2.;return o
 mf.get_occ=occupations;mf.max_cycle=150;mf.conv_tol=1e-11;mf.conv_tol_grad=1e-8;mf.diis_space=12
 checkpoint=out/'scf_checkpoint.npz'
 if checkpoint.exists():
  with np.load(checkpoint) as z:dm=z['density'].copy()
 else:dm=2*s.coefficients[:,:a.no]@s.coefficients[:,:a.no].T
 history=json.loads((out/'history.json').read_text()) if (out/'history.json').exists() else []
 def callback(env):
  temporary=checkpoint.with_suffix('.tmp.npz');np.savez_compressed(temporary,density=env['dm']);temporary.replace(checkpoint);history.append(dict(cycle=len(history),energy=float(env['e_tot']),gradient_norm=float(env['norm_gorb'])));write(out/'history.json',history);print(name,history[-1],flush=True)
 mf.callback=callback;mf.kernel(dm0=dm);assert mf.converged,'Compact anchor did not converge; checkpoint preserved'
 p=mf.make_rdm1();f,e=a.evaluate(p);levels,c=eigh(f,s.overlap);fixed=2*c[:,:a.no]@c[:,:a.no].T;residual=float(np.max(abs((c.T@f@c)[:a.no,a.no:])));assert residual<1e-8 and np.max(abs(fixed-p))<1e-7
 np.savez_compressed(out/'anchor.npz',coefficients=c,density=p,fock=f,orbital_energies=levels,energy_up_to_fixed_core_constant=e)
 rng=np.random.default_rng(41);direction=rng.normal(size=p.shape);direction=.5*(direction+direction.T);direction/=np.linalg.norm(direction);eps=1e-5;error=abs((a.evaluate(p+eps*direction)[1]-a.evaluate(p-eps*direction)[1])/(2*eps)-np.sum(f*direction));assert error<2e-7
 write(out/'certificate.json',dict(host=socket.getfqdn(),wall_seconds=time.monotonic()-start,residual=residual,energy_fock_derivative_error=error,RI_JK_auxbasis='def2-universal-jkfit',XC='CAM-B3LYP',grid_level=3,no_excitation_offset=True,energy_labels_read=False,status='Frozen-core compact anchor complete; response calibration and OO states still required'))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--wait-for-integrals',action='store_true');args=p.parse_args();out=HERE/'pigment_compact_reference'/args.name;out.mkdir(parents=True,exist_ok=True);lock=out/'worker.lock';fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL);os.write(fd,str(os.getpid()).encode());os.close(fd)
 try:
  while not (HERE/'pigment_integrals'/args.name/'long_range_complete.json').exists():
   if not args.wait_for_integrals:raise RuntimeError('Projected integral checkpoint incomplete')
   time.sleep(20)
  with threadpool_limits(8):run(args.name)
 finally:lock.unlink(missing_ok=True)
