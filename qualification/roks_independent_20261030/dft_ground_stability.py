"""Check whether HNO triplet instability also occurs in the parent RKS reference.

Reuse the converged SCF orbitals. This is the full PBE0/aug-cc-pVDZ reference
(a=.25), distinct from the compact frozen-response model using lambda=.9.
"""
from remote_bootstrap import BASE,HOST
from pathlib import Path
import sys,numpy as np
from pyscf import scf,dft,lib
from pyscf.scf import stability
from threadpoolctl import threadpool_limits
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import plan,write,digest
lib.num_threads(2)
case=next(c for c in plan() if c['name']=='nitroxyl');folder=Path(case['reference']);out=BASE/'results/response_ablation/dft_ground_stability.json'
if out.exists():raise SystemExit('completed reference stability reused')
mol=scf.chkfile.load_mol(str(folder/'backend.chk'));mf=dft.RKS(mol);mf.xc='PBE0';mf.verbose=0;mf.small_rho_cutoff=0;mf.max_memory=12000;mf.grids.level=3
with np.load(folder/'reference.npz') as f:mf.mo_coeff=f['coefficients'];mf.mo_occ=f['occupations'];mf.mo_energy=f['orbital_energies_hartree']
mf.converged=True;mf._eri=mol.intor('int2e',aosym='s8')
with threadpool_limits(limits=2,user_api='blas'):
 _,_,hop,diag=stability._gen_hop_rhf_external(mf,with_symmetry=False)
 rng=np.random.default_rng(20261030);x=rng.normal(size=len(diag));x/=np.linalg.norm(x)
 checkpoint=out.with_suffix('.npz')
 if checkpoint.exists():
  with np.load(checkpoint) as saved:x=saved['eigenvector']
 def precondition(dx,e,x0):
  d=diag-e;d=np.where(abs(d)<1e-8,1e-8,d);return dx/d
 e,v=lib.davidson(hop,x,precondition,tol=1e-14,max_cycle=100,max_space=30,nroots=1,verbose=0)
 residual=float(np.linalg.norm(hop(v)-e*v));np.savez_compressed(checkpoint,eigenvector=v,eigenvalue=e,residual=residual);assert residual<1e-7
 report=dict(host=HOST,molecule='nitroxyl',source_sha256=digest(__file__),reference_sha256=digest(folder/'reference.npz'),lowest_triplet_hessian_over4_hartree=float(e),residual=residual,stable=bool(e>=-1e-8),SCF_repeated=False,scope='parent full-basis PBE0 RKS, no lambda response modification')
write(out,report);print(report,flush=True)
