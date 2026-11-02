"""Independent molecular derivatives, coherent spin and functional-family tests."""
import socket,runpy,json,time,sys
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from pyscf import gto,dft,scf
from threadpoolctl import threadpool_limits
assert socket.getfqdn().lower()=='mac-2.homenet.telecomitalia.it'
runpy.run_path('/Users/vincenzobarone/MOUSE-oo-response-20261101/source/qualification/remote_bootstrap.py')
from matrix_pyscf import optimize_roks,optimize_deltascf
from matrix_pyscf.roks import _singlet_overlap
root=Path(__file__).resolve().parent
cases=[('formaldehyde','PBE0','6-31g','roks'),('water','CAM-B3LYP','sto-3g','roks'),
       ('water','LDA,VWN','sto-3g','roks'),('water','PBE0','sto-3g','deltascf')]
selected=sys.argv[1:]
with threadpool_limits(2):
 for name,method,basis,kind in cases:
  key=(name+'_'+method+'_'+kind).replace(',','_');out=root/(key+'_validation.json')
  if selected and key not in selected:continue
  if out.exists():print('REUSE',key,flush=True);continue
  atom='C 0 0 0; O 0 0 1.21; H 0 .94 -.59; H 0 -.94 -.59' if name=='formaldehyde' else 'O 0 0 0; H 0 -.757 .587; H 0 .757 .587'
  mol=gto.M(atom=atom,basis=basis,verbose=0);ground=dft.RKS(mol);ground.xc=method;ground.grids.level=2;ground.small_rho_cutoff=0
  ground.conv_tol=1e-12;ground.kernel();assert ground.converged
  mf=dft.UKS(mol);mf.xc=method;mf.grids.level=2;mf.small_rho_cutoff=0
  c=ground.mo_coeff;occ=np.array([ground.mo_occ/2]*2);n=mol.nelectron//2;occ[0,n-1]=0;occ[0,n]=1
  solver=optimize_roks if kind=='roks' else optimize_deltascf
  initial=c if kind=='roks' else np.array([c,c]);checkpoint=root/(key+'_checkpoint.npz');start=time.monotonic()
  result=solver(mf,initial,occ,ground_energy_hartree=ground.e_tot,include_hessian=True,
      orbital_tolerance=1e-10,checkpoint=checkpoint,restart=checkpoint if checkpoint.exists() else None)
  saved_c=result.common_coefficients if kind=='roks' else result.coefficients
  energies={};gradients={};rng=np.random.default_rng(22);direction=rng.normal(size=mol.natm*3);direction/=np.linalg.norm(direction)
  for step in [4e-5,2e-5]:
   for sign in [-1,1]:
    xyz=mol.atom_coords()+sign*step*direction.reshape(-1,3);othermol=mol.copy().set_geom_(xyz,unit='Bohr')
    other=dft.UKS(othermol);other.xc=method;other.grids.level=2;other.small_rho_cutoff=0
    projected=[]
    for a in ([saved_c] if kind=='roks' else saved_c):
     z=scf.addons.project_mo_nr2nr(mol,a,othermol);v,u=eigh(z.T@other.get_ovlp()@z);projected.append(z@(u*v**-.5)@u.T)
    trial=solver(other,projected[0] if kind=='roks' else np.array(projected),occ,include_gradient=True,orbital_tolerance=1e-11)
    energies[step*sign]=trial.energy_hartree;gradients[step*sign]=trial.gradient_hartree_per_bohr.ravel()
  ga=(energies[4e-5]-energies[-4e-5])/8e-5;gb=(energies[2e-5]-energies[-2e-5])/4e-5
  ha=(gradients[4e-5]-gradients[-4e-5])/8e-5;hb=(gradients[2e-5]-gradients[-2e-5])/4e-5
  ge=abs((4*gb-ga)/3-result.gradient_hartree_per_bohr.ravel()@direction)
  he=float(np.max(abs((4*hb-ha)/3-result.hessian_hartree_per_bohr2@direction)))
  assert ge<2e-7,ge
  assert he<2e-6,he
  spin_norm=None;coherent_error=None
  if kind=='roks':
   # Apply the actual spin-raising operator to the two determinant bitstrings.
   occupied=[np.flatnonzero(o>.5) for o in occ];width=len(occ[0]);amplitudes={}
   for a,b in [occupied,occupied[::-1]]:
    bits=sum(1<<int(i) for i in a)+sum(1<<int(i+width) for i in b)
    for orbital in range(width):
     source=width+orbital;target=orbital
     if not bits>>source&1 or bits>>target&1:continue
     phase=(-1)**((bits&((1<<source)-1)).bit_count());removed=bits^(1<<source)
     phase*=(-1)**((removed&((1<<target)-1)).bit_count());final=removed|(1<<target)
     amplitudes[final]=amplitudes.get(final,0.)+phase/np.sqrt(2)
   spin_norm=sum(abs(v)**2 for v in amplitudes.values());assert spin_norm==0.
   from matrix_apoc import singlet_csf_overlap
   from matrix_pyscf._oo_surface import OrbitalSurface
   surface=OrbitalSurface(mf,saved_c,occ,common=True);v=rng.normal(size=surface.dimension);v*=.05/np.linalg.norm(v)
   rotated=surface.rotate(saved_c,v)
   from dataclasses import replace
   candidate=replace(result,common_coefficients=rotated)
   actual=singlet_csf_overlap(result.to_apoc(mf),candidate.to_apoc(mf),mf.get_ovlp())[0]
   coherent_error=abs(actual-_singlet_overlap(saved_c,rotated,occ,mf.get_ovlp()));assert coherent_error<1e-12
  report=dict(host=socket.getfqdn(),molecule=name,method=method,basis=basis,kind=kind,
      energy_hartree=result.energy_hartree,excitation_ev=result.excitation_energy_hartree*27.211386245988,
      residual=result.orbital_residual,iterations=result.iterations,spin_squared=result.spin_squared,
      gradient_error=ge,Hessian_vector_error=he,spin_raising_norm_squared=spin_norm,
      coherent_overlap_error=coherent_error,response=result.response_certificate,seconds=time.monotonic()-start)
  np.savez_compressed(root/(key+'_derivatives.npz'),gradient=result.gradient_hartree_per_bohr,hessian=result.hessian_hartree_per_bohr2)
  out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
