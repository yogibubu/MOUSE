"""Auxiliary-density XC response with explicit full-grid curvature controls.

RI-V coefficients use the same recorded JK-fitting basis. The reference
charge density and f_xc are evaluated in the original embedded AO image.
Only the response density is resolved in auxiliary Gaussian functions.
This is an explicit numerical approximation, not an energy fit or offset.
"""
from pathlib import Path
import json,time,socket
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
import numpy as np
from scipy.linalg import cholesky,solve_triangular,cho_solve
from pyscf import df,gto
from pigment_reference_embedding import write
from pigment_block_matmul import product

def prepare(case,p,i,j,w,out,tolerance=1e-5,require_accuracy=True):
 out=Path(out)/'ri_xc';out.mkdir(parents=True,exist_ok=True);nf=len(i);aux=df.addons.make_auxmol(case.mol,'def2-universal-jkfit');na=aux.nao_nr();assert na==len(case.a),'RI-V metric decomposition needs the full-rank Coulomb factor chart'
 coefficient_path=out/'density_coefficients.npy'
 if not coefficient_path.exists():
  metric=aux.intor('int2c2e');chol=cholesky(metric,lower=True);lp=np.array(case.a[:,i,j]*w[None,:],order='C');coefficients=solve_triangular(chol.T,lp,lower=False);np.save(coefficient_path,coefficients);del metric,chol,lp,coefficients
 # Impose exact charge, dipole and quadrupole moments in the same Coulomb
 # fit. These are analytic Gaussian integrals, never excitation labels.
 constrained=out/'density_coefficients_multipole.npy'
 if not constrained.exists():
  metric=aux.intor('int2c2e');chol=cholesky(metric,lower=True);coefficients=np.load(coefficient_path)
  origin=case.mol.atom_coords().mean(axis=0);constant=gto.M(atom=[('Ghost-H',origin)],unit='Bohr',basis={'H':[[0,[1e-18,1.]]]},spin=0,charge=0,verbose=0);normalization=constant.eval_gto('GTOval_sph',origin[None])[0,0]
  Q=[gto.intor_cross('int1e_ovlp',aux,constant)[:,0]/normalization];target=[case.lift.T@case.mol.intor('int1e_ovlp')@case.lift]
  with aux.with_common_orig(origin),constant.with_common_orig(origin),case.mol.with_common_orig(origin):
   dipole=gto.intor_cross('int1e_r',aux,constant)[:,:,0]/normalization;quadrupole=gto.intor_cross('int1e_rr',aux,constant)[:,:,0]/normalization
   for component in range(3):Q.append(dipole[component])
   for component in [0,4,8,1,2,5]:Q.append(quadrupole[component])
   dipole=case.mol.intor('int1e_r');quadrupole=case.mol.intor('int1e_rr')
   for component in range(3):target.append(case.lift.T@dipole[component]@case.lift)
   for component in [0,4,8,1,2,5]:target.append(case.lift.T@quadrupole[component]@case.lift)
  Q=np.asarray(Q);target=np.asarray(target)[:,i,j]*w[None,:];scales=np.maximum(np.linalg.norm(Q,axis=1),1e-12);Q/=scales[:,None];target/=scales[:,None];Y=cho_solve((chol,True),Q.T);delta=target-Q@coefficients;coefficients+=Y@np.linalg.solve(Q@Y,delta);moment_error=float(np.max(abs(target-Q@coefficients)));assert moment_error<1e-8;np.save(constrained,coefficients);write(out/'moment_certificate.json',dict(charge_dipole_quadrupole_constraints=10,maximum_normalized_moment_error=moment_error,energy_labels_read=False))
 coefficient_path=constrained
 coefficients=np.load(coefficient_path,mmap_mode='r');n=case.n;rng=np.random.default_rng(71);directions=rng.normal(size=(3,n,n));directions=.5*(directions+directions.transpose(0,2,1));directions/=np.linalg.norm(directions,axis=(1,2))[:,None,None];q=directions[:,i,j]*w[None,:];resolved=coefficients@q.T;direct_sum=np.zeros((2,3));fitted_sum=np.zeros((2,3));journal=out/'chunks_multipole';journal.mkdir(exist_ok=True);tags=[]
 for block,path in enumerate(case.grid):
  with np.load(path) as z:phi=z['phi'];core=z['core'];weight=z['weight'];coordinates=z['coordinates']
  rho=.5*(core+case.rho(phi,p));fxc=case.mf._numint.eval_xc_eff('CAM-B3LYP',np.array([rho,rho]),deriv=2,xctype='GGA')[2];fc=.25*(fxc[0,:,0]+fxc[0,:,1]+fxc[1,:,0]+fxc[1,:,1]);fs=.25*(fxc[0,:,0]-fxc[0,:,1]-fxc[1,:,0]+fxc[1,:,1])
  for lo in range(0,len(weight),512):
   hi=min(len(weight),lo+512);tag=f'{block:04d}_{lo:06d}';meta=journal/(tag+'.json');tags.append(tag)
   if meta.exists():
    d=json.loads(meta.read_text());direct_sum+=d['direct'];fitted_sum+=d['fitted'];continue
   auxiliary=aux.eval_gto('GTOval_sph_deriv1',coordinates[lo:hi]);active=np.flatnonzero(np.max(abs(auxiliary),axis=(0,1))>1e-12);feature=auxiliary[:,:,active].transpose(0,2,1);flat=feature.transpose(1,0,2).reshape(len(active),-1);ww=weight[lo:hi];direct=np.array([case.rho(phi[:,lo:hi],d) for d in directions]);fitted=np.einsum('kga,ai->ikg',auxiliary,resolved,optimize=True);dc=np.empty((2,3));dfit=np.empty((2,3))
   for channel,fx in enumerate([fc[:,:,lo:hi],fs[:,:,lo:hi]]):
    weighted=(np.einsum('ijg,jag->aig',fx,feature,optimize=True)*ww[None,None,:]).reshape(len(active),-1);matrix=flat@weighted.T;np.save(journal/(tag+f'_{channel}.npy'),matrix);dc[channel]=np.einsum('aig,ijg,ajg,g->a',direct,fx,direct,ww,optimize=True);dfit[channel]=np.einsum('aig,ijg,ajg,g->a',fitted,fx,fitted,ww,optimize=True)
   np.save(journal/(tag+'_ids.npy'),active);write(meta,dict(direct=dc.tolist(),fitted=dfit.tolist(),auxiliary_functions=len(active)));direct_sum+=dc;fitted_sum+=dfit
  print(case.name,'auxiliary XC quadrature',block+1,'/',len(case.grid),flush=True)
 error=float(np.max(abs(direct_sum-fitted_sum)));report=dict(auxbasis='def2-universal-jkfit',auxiliary_functions=na,native_pair_features=nf,direct_curvatures=direct_sum.tolist(),fitted_curvatures=fitted_sum.tolist(),maximum_curvature_error_hartree=error,required_tolerance_hartree=tolerance,energy_labels_read=False,approximation='RI-V response density with exact charge/dipole/quadrupole constraints; full embedded density used for f_xc');write(out/'quadrature_certificate.json',report);assert not require_accuracy or error<tolerance,('Auxiliary XC response not qualified',error,tolerance)
 results=[]
 for channel in range(2):
  matrix=np.zeros((na,na))
  for tag in tags:
   ids=np.load(journal/(tag+'_ids.npy'));matrix[np.ix_(ids,ids)]+=np.load(journal/(tag+f'_{channel}.npy'))
  transformed_path=out/f'weighted_coefficients_multipole_{channel}.npy'
  if not transformed_path.exists():np.save(transformed_path,matrix@coefficients)
  results.append(product(coefficient_path,transformed_path,out/f'kernel_multipole_{channel}.npy'))
 return results,report
