"""Electronic qualification: run only on an explicitly verified compute host."""
import socket,sys,importlib.util,json,time
from pathlib import Path
import numpy as np
host=socket.getfqdn().lower()
assert host=='mac-2.homenet.telecomitalia.it',host
from pyscf import gto,dft,scf,lib
from pyscf.soscf.newton_ah import gen_g_hop_uhf
from threadpoolctl import threadpool_limits
spec=importlib.util.spec_from_file_location('dft_adapter',Path(__file__).with_name('deltascf.py'))
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
root=Path(__file__).resolve().parent;lib.num_threads(2)
report=dict(host=host,finite_differences_in_production=False,TDA_calls=0,cases=[])
with threadpool_limits(2):
 for method in ['HF','PBE0']:
  saved=root/(method+'_validation.json')
  if saved.exists():report['cases'].append(json.loads(saved.read_text()));continue
  mol=gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',basis='sto-3g',verbose=0)
  ground=scf.RHF(mol) if method=='HF' else dft.RKS(mol)
  if method!='HF':ground.xc=method;ground.grids.level=1
  ground.conv_tol=1e-12;ground.kernel();assert ground.converged
  mf=scf.UHF(mol) if method=='HF' else dft.UKS(mol)
  if method!='HF':mf.xc=method;mf.grids.level=1
  c=np.array([ground.mo_coeff]*2);occ=np.zeros((2,len(c[0])));occ[:,:5]=1;occ[0,4]=0;occ[0,5]=1
  rng=np.random.default_rng(8);dimension=sum(int(o.sum())*(len(o)-int(o.sum())) for o in occ)
  direction=rng.normal(size=dimension);direction/=np.linalg.norm(direction)
  trial=module._rotate(c,occ,.03*direction)
  dm=module._density(trial,occ);f=mf.get_hcore()+mf.get_veff(dm=dm)
  g,action,diag=gen_g_hop_uhf(mf,trial,occ,fock_ao=f,with_symmetry=False)
  step=1e-5
  def residual(x):
   y=module._rotate(trial,occ,x*direction);p=module._density(y,occ)
   return gen_g_hop_uhf(mf,y,occ,fock_ao=mf.get_hcore()+mf.get_veff(dm=p),with_symmetry=False)[0]
  derivative_error=float(np.max(abs((residual(step)-residual(-step))/(2*step)-action(direction))))
  assert derivative_error<2e-7,derivative_error
  checkpoint=root/(method+'_checkpoint.npz');start=time.monotonic()
  result=module.optimize_deltascf(mf,trial,occ,ground_energy_hartree=ground.e_tot,
      checkpoint=checkpoint,include_gradient=True,orbital_tolerance=1e-8,maximum_iterations=100)
  restart=module.optimize_deltascf(mf,trial,occ,restart=checkpoint,include_gradient=True,orbital_tolerance=1e-8)
  assert restart.iterations==0
  assert abs(result.energy_hartree-restart.energy_hartree)<1e-11
  assert np.max(abs(mf.get_ovlp()-ground.get_ovlp()))==0
  # Independent scalar-energy nuclear check: project converged occupied spaces
  # into displaced AO bases and solve stationarity again, never a new ground SCF.
  numeric=[]
  for displacement in [-2e-4,2e-4]:
   shifted=mol.copy();xyz=mol.atom_coords();xyz[1,2]+=displacement;shifted.set_geom_(xyz,unit='Bohr')
   other=scf.UHF(shifted) if method=='HF' else dft.UKS(shifted)
   if method!='HF':other.xc=method;other.grids.level=1
   from scipy.linalg import eigh
   projected=[]
   for spin in range(2):
    z=scf.addons.project_mo_nr2nr(mol,result.coefficients[spin],shifted)
    values,u=eigh(z.T@other.get_ovlp()@z);projected.append(z@(u*values**-.5)@u.T)
   r=module.optimize_deltascf(other,np.array(projected),occ,orbital_tolerance=1e-9)
   numeric.append(r.energy_hartree)
  gradient_error=abs((numeric[1]-numeric[0])/4e-4-result.gradient_hartree_per_bohr[1,2])
  assert gradient_error<2e-5,gradient_error
  # Reference PySCF IMOM: same determinant, functional and geometry.
  mom=scf.UHF(mol) if method=='HF' else dft.UKS(mol)
  if method!='HF':mom.xc=method;mom.grids.level=1
  scf.addons.mom_occ(mom,trial,occ);mom.conv_tol=1e-10;mom.max_cycle=150;mom.kernel(dm0=module._density(trial,occ))
  row=dict(method=method,energy_hartree=result.energy_hartree,excitation_ev=result.excitation_energy_hartree*27.211386245988,
      residual=result.orbital_residual,iterations=result.iterations,spin_squared=result.spin_squared,
      target_overlap=result.reference_overlap,orbital_Hessian_action_error=derivative_error,
      nuclear_gradient_error=gradient_error,restart_energy_error=abs(result.energy_hartree-restart.energy_hartree),
      MOM_converged=bool(mom.converged),MOM_energy_hartree=float(mom.e_tot),seconds=time.monotonic()-start)
  saved.write_text(json.dumps(row,indent=2)+'\n');report['cases'].append(row);print(json.dumps(row),flush=True)
(root/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
