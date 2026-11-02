"""Second-molecule and public APOC qualification, remote execution only."""
from pathlib import Path
import json,runpy,socket,importlib.util,sys
import numpy as np
from threadpoolctl import threadpool_limits
from pyscf import gto,dft,scf
from pyscf.soscf.newton_ah import gen_g_hop_uhf
assert socket.getfqdn().lower()=='mac-2.homenet.telecomitalia.it'
root=Path(__file__).resolve().parent
runpy.run_path('/Users/vincenzobarone/MOUSE-oo-response-20261101/source/qualification/remote_bootstrap.py')
spec=importlib.util.spec_from_file_location('dft_adapter',root/'deltascf.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
with threadpool_limits(2):
 mol=gto.M(atom='C 0 0 0; O 0 0 1.21; H 0 .94 -.59; H 0 -.94 -.59',basis='6-31g',verbose=0)
 ground=dft.RKS(mol);ground.xc='PBE0';ground.grids.level=2;ground.conv_tol=1e-12;ground.kernel();assert ground.converged
 mf=dft.UKS(mol);mf.xc=ground.xc;mf.grids.level=2
 c=np.array([ground.mo_coeff]*2);occ=np.array([ground.mo_occ/2]*2);n=mol.nelectron//2;occ[0,n-1]=0;occ[0,n]=1
 result=module.optimize_deltascf(mf,c,occ,ground_energy_hartree=ground.e_tot,include_gradient=True,checkpoint=root/'formaldehyde_checkpoint.npz',orbital_tolerance=1e-8)
 fock=mf.get_hcore()+mf.get_veff(dm=module._density(result.coefficients,occ));g,action,_=gen_g_hop_uhf(mf,result.coefficients,occ,fock_ao=fock,with_symmetry=False)
 # A small-molecule qualification only; production remains matrix-free.
 hessian=np.column_stack([action(x) for x in np.eye(len(g))]);eig=np.linalg.eigvalsh((hessian+hessian.T)/2)
 assert eig[0]<-1e-4
 exported=result.to_apoc(mf)
 from matrix_apoc import determinant_overlap
 overlap,phase=determinant_overlap(exported,exported,mf.get_ovlp());assert abs(overlap-1)<1e-10
 changed=dft.UKS(mol);changed.xc='BLYP';changed.grids.level=2
 try:module.optimize_deltascf(changed,c,occ,restart=root/'formaldehyde_checkpoint.npz')
 except ValueError as exc:assert 'mismatch' in str(exc)
 else:raise AssertionError('wrong-functional restart accepted')
 energies=[]
 from scipy.linalg import eigh
 for displacement in [-2e-4,2e-4]:
  xyz=mol.atom_coords();xyz[1,2]+=displacement;othermol=mol.copy().set_geom_(xyz,unit='Bohr')
  other=dft.UKS(othermol);other.xc=mf.xc;other.grids.level=2
  projected=[]
  for spin in range(2):
   z=scf.addons.project_mo_nr2nr(mol,result.coefficients[spin],othermol);values,u=eigh(z.T@other.get_ovlp()@z);projected.append(z@(u*values**-.5)@u.T)
  r=module.optimize_deltascf(other,np.array(projected),occ,orbital_tolerance=1e-9)
  energies.append(r.energy_hartree)
 error=abs((energies[1]-energies[0])/4e-4-result.gradient_hartree_per_bohr[1,2]);assert error<2e-6,error
 report=dict(host=socket.getfqdn(),molecule='formaldehyde',basis='6-31g',xc=mf.xc,energy_hartree=result.energy_hartree,
 excitation_ev=result.excitation_energy_hartree*27.211386245988,iterations=result.iterations,residual=result.orbital_residual,
 spin_squared=result.spin_squared,nuclear_gradient_error=error,negative_orbital_modes=int(np.sum(eig< -1e-6)),
 minimum_orbital_Hessian_eigenvalue=float(eig[0]),APOC_self_overlap=overlap,wrong_functional_restart_rejected=True,
 nuclear_Hessian_implemented=False,spin_pure_singlet=False,finite_differences_in_production=False)
 (root/'extended_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
