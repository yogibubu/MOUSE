"""Resume the same reference root with cached completed operator products."""
from remote_bootstrap import BASE,HOST
from pathlib import Path
import sys,numpy as np
from pyscf import scf,dft,lib
from threadpoolctl import threadpool_limits
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import write,digest
out=BASE/'references_C/pyrazine';final=out/'backend_tda.npz'
if final.exists():raise SystemExit('completed reference root reused')
with np.load(out/'tda_progress.npz') as f:assert str(f['reference_sha256'])==digest(out/'reference.npz');xs=f['xs'];ax=f['ax'];old_e=float(f['e'][0])
mol=scf.chkfile.load_mol(str(out/'backend.chk'));mf=dft.RKS(mol);mf.xc='PBE0';mf.verbose=0;mf.small_rho_cutoff=0;mf.grids.level=3;mf.max_memory=40000
with np.load(out/'reference.npz') as f:mf.mo_coeff=f['coefficients'];mf.mo_occ=f['occupations'];mf.mo_energy=f['orbital_energies_hartree'];mf.e_tot=float(f['energy_hartree'])
mf.converged=True;lib.num_threads(4);td=mf.TDA();td.verbose=0;vind,diag=td.gen_vind();cached=[0]
def hop(x):
 proj=xs@x;remainder=x-proj@xs
 if np.linalg.norm(remainder)<1e-12:cached[0]+=1;return proj@ax
 return vind(remainder[None])[0]+proj@ax

def precondition(dx,e,x0):
 d=diag-e;d=np.where(abs(d)<1e-8,1e-8,d);return dx/d
with threadpool_limits(limits=2,user_api='blas'):
 e,v=lib.davidson(hop,xs[0],precondition,tol=1e-16,lindep=1e-24,max_space=30,max_cycle=100,verbose=0)
 residual=float(np.linalg.norm(vind(v[None])[0]-e*v))
 np.savez_compressed(out/'tda_resumed_eigenpair.npz',energy=e,amplitude=v,residual=residual)
 assert residual<1e-8 and abs(e-old_e)<1e-6,(residual,e-old_e)
 np.savez_compressed(final,amplitudes=v[None]/np.linalg.norm(v),energies_hartree=np.array([e]))
 write(out/'tda_solver_recovery.json',dict(host=HOST,source_sha256=digest(__file__),eigenvalue_hartree=float(e),old_eigenvalue_hartree=old_e,residual=residual,
  completed_matvecs_preserved=len(xs),cached_operator_calls=cached[0],SCF_repeated=False,physical_operator_changed=False,reference_root_changed=False,requested_tolerance=1e-8,lindep=1e-24))
print('REFERENCE TDA RESUMED',float(e),residual,'cached',cached[0],flush=True)
