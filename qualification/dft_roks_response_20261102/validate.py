"""Remote analytic ROKS qualification, resumable by completed cases."""
import socket,runpy,json,time,sys
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
from pyscf import gto,dft,scf
assert socket.getfqdn().lower()=='mac-2.homenet.telecomitalia.it'
runpy.run_path('/Users/vincenzobarone/MOUSE-oo-response-20261101/source/qualification/remote_bootstrap.py')
from matrix_pyscf import optimize_roks
from matrix_pyscf._oo_surface import OrbitalSurface
root=Path(__file__).resolve().parent;methods=sys.argv[1:] or ['HF','PBE0']
with threadpool_limits(2):
 for method in methods:
  out=root/(method+'_validation.json')
  if out.exists():print('REUSE',out);continue
  mol=gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',basis='sto-3g',verbose=0)
  ground=scf.RHF(mol) if method=='HF' else dft.RKS(mol)
  mf=scf.UHF(mol) if method=='HF' else dft.UKS(mol)
  if method!='HF':
   ground.xc=method;ground.grids.level=1;ground.small_rho_cutoff=0
   mf.xc=method;mf.grids.level=1;mf.small_rho_cutoff=0
  ground.conv_tol=1e-12;ground.kernel();assert ground.converged
  c=ground.mo_coeff;occ=np.array([ground.mo_occ/2]*2);occ[0,4]=0;occ[0,5]=1
  start=time.monotonic();checkpoint=root/(method+'_checkpoint.npz')
  result=optimize_roks(mf,c,occ,ground_energy_hartree=ground.e_tot,checkpoint=checkpoint,
                       restart=checkpoint if checkpoint.exists() else None,orbital_tolerance=1e-10,include_hessian=True)
  np.savez_compressed(root/(method+'_derivatives.npz'),gradient=result.gradient_hartree_per_bohr,hessian=result.hessian_hartree_per_bohr2)
  # Independent scalar energy and gradient checks along two nuclear directions.
  rng=np.random.default_rng(9);directions=[rng.normal(size=9),np.array([0,0,0,1.,0,0,-1.,0,0])]
  checks=[]
  for direction in directions:
   direction/=np.linalg.norm(direction);values={};gradients={}
   for h in [4e-5,2e-5]:
    for sign in [-1,1]:
     geometry=mol.atom_coords()+sign*h*direction.reshape(-1,3)
     othermol=mol.copy().set_geom_(geometry,unit='Bohr');other=scf.UHF(othermol) if method=='HF' else dft.UKS(othermol)
     if method!='HF':other.xc=mf.xc;other.grids.level=mf.grids.level;other.small_rho_cutoff=0
     z=scf.addons.project_mo_nr2nr(mol,result.common_coefficients,othermol)
     v,u=eigh(z.T@other.get_ovlp()@z);z=z@(u*v**-.5)@u.T
     trial=optimize_roks(other,z,occ,orbital_tolerance=1e-11,include_gradient=True)
     values[sign*h]=trial.energy_hartree;gradients[sign*h]=trial.gradient_hartree_per_bohr.ravel()
   a=(values[4e-5]-values[-4e-5])/8e-5;b=(values[2e-5]-values[-2e-5])/4e-5
   ga=(gradients[4e-5]-gradients[-4e-5])/8e-5;gb=(gradients[2e-5]-gradients[-2e-5])/4e-5
   gradient_error=abs((4*b-a)/3-result.gradient_hartree_per_bohr.ravel()@direction)
   hessian_error=float(np.max(abs((4*gb-ga)/3-result.hessian_hartree_per_bohr2@direction)))
   assert gradient_error<2e-7,gradient_error
   assert hessian_error<2e-6,hessian_error
   checks.append(dict(gradient_error=gradient_error,Hessian_vector_error=hessian_error))
  # Off-stationary analytic moving-frame orbital action.
  surface=OrbitalSurface(mf,result.common_coefficients,occ,common=True);v=rng.normal(size=surface.dimension);v/=np.linalg.norm(v)
  trialc=surface.rotate(result.common_coefficients,.025*v);energy,g,action,*unused=surface.evaluate(trialc)
  delta=1e-5;fd=(surface.evaluate(surface.rotate(trialc,delta*v))[1]-surface.evaluate(surface.rotate(trialc,-delta*v))[1])/(2*delta)
  action_error=float(np.max(abs(fd-action(v))));assert action_error<2e-7,action_error
  exported=result.to_apoc(mf)
  from matrix_apoc import singlet_csf_overlap
  score=singlet_csf_overlap(exported,exported,mf.get_ovlp());assert abs(score[0]-1)<1e-10
  replay=optimize_roks(mf,c,occ,restart=checkpoint,orbital_tolerance=1e-10)
  assert replay.iterations==0 and abs(replay.energy_hartree-result.energy_hartree)<1e-11
  report=dict(host=socket.getfqdn(),method=method,basis='sto-3g',molecule='water',energy_hartree=result.energy_hartree,
              excitation_ev=result.excitation_energy_hartree*27.211386245988,spin_squared=result.spin_squared,iterations=result.iterations,
              residual=result.orbital_residual,checks=checks,orbital_action_error=action_error,APOC_CSF_self_overlap=score,
              restart_iterations=replay.iterations,response=result.response_certificate,seconds=time.monotonic()-start,
              energy_scope='Common-orbital 2 E_M - E_T functional approximation, not exact DFT spin projection')
  out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
