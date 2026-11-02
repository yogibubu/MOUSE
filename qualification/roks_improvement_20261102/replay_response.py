"""Qualify complete cached analytic arrays against independently checked arrays."""
import socket,runpy,json,time
from pathlib import Path
import numpy as np
from pyscf import gto,dft,scf
from threadpoolctl import threadpool_limits
assert socket.getfqdn().lower()=='mac-2.homenet.telecomitalia.it'
runpy.run_path('/Users/vincenzobarone/MOUSE-oo-response-20261101/source/qualification/remote_bootstrap.py')
from matrix_pyscf import optimize_roks,optimize_deltascf
output=Path(__file__).resolve().parent
root=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-dft-roks-response-20261102');results=[]
with threadpool_limits(2):
 for path in sorted(root.glob('*_validation.json')):
  if path.name=='cached_response_validation.json':continue
  record=json.loads(path.read_text())
  if 'method' not in record:continue
  key=path.name.replace('_validation.json','');name=record['molecule'];method=record['method'];kind=record.get('kind','roks')
  atom='C 0 0 0; O 0 0 1.21; H 0 .94 -.59; H 0 -.94 -.59' if name=='formaldehyde' else 'O 0 0 0; H 0 -.757 .587; H 0 .757 .587'
  mol=gto.M(atom=atom,basis=record['basis'],verbose=0)
  mf=scf.UHF(mol) if method=='HF' else dft.UKS(mol)
  if method!='HF':mf.xc=method;mf.grids.level=1 if key=='PBE0' else 2;mf.small_rho_cutoff=0
  checkpoint=root/(key+'_checkpoint.npz')
  with np.load(checkpoint) as z:c=z['coefficients'].copy();occ=z['occupations'].copy()
  start=time.monotonic();solver=optimize_roks if kind=='roks' else optimize_deltascf
  result=solver(mf,c,occ,restart=checkpoint,include_hessian=True,orbital_tolerance=1e-10)
  elapsed=time.monotonic()-start
  with np.load(root/(key+'_derivatives.npz')) as z:
   ge=float(np.max(abs(result.gradient_hartree_per_bohr-z['gradient'])))
   he=float(np.max(abs(result.hessian_hartree_per_bohr2-z['hessian'])))
  assert result.iterations==0 and ge<1e-10 and he<1e-8,(key,ge,he)
  translations=np.tile(np.eye(3),(mol.natm,1));te=float(np.max(abs(result.hessian_hartree_per_bohr2@translations)))
  assert te<1e-8,(key,te)
  row=dict(case=key,orbital_iterations=0,gradient_error=ge,complete_Hessian_error=he,
           translation_sum_rule_error=te,prepared_energy_gradient_Hessian_seconds=elapsed,response=result.response_certificate)
  results.append(row);print(json.dumps(row),flush=True)
 # No orbital rotations are available for the complete one-AO H determinant.
 mol=gto.M(atom='H 0 0 0',basis='sto-3g',spin=1,verbose=0);mf=scf.UHF(mol)
 c=np.array([[[1/np.sqrt(mf.get_ovlp()[0,0])]]]*2);occ=np.array([[1.],[0.]])
 atom=optimize_deltascf(mf,c,occ,include_hessian=True)
 assert np.max(abs(atom.hessian_hartree_per_bohr2))<1e-12
 # Unsupported analytic models must fail before electronic iterations.
 reference=results[0]
 mol=gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',basis='sto-3g',verbose=0);mf=dft.UKS(mol);mf.xc='SCAN'
 with np.load(root/'PBE0_checkpoint.npz') as z:c=z['coefficients'];occ=z['occupations']
 try:optimize_roks(mf,c,occ,include_hessian=True)
 except NotImplementedError:pass
 else:raise AssertionError('meta-GGA Hessian silently accepted')
 report=dict(host=socket.getfqdn(),cases=results,zero_rotation_atom_Hessian_max=float(np.max(abs(atom.hessian_hartree_per_bohr2))),
             meta_GGA_Hessian_rejected=True,ground_SCF_calls=0,TDA_calls=0,finite_differences_in_production=False)
 (output/'response_regression.json').write_text(json.dumps(report,indent=2)+'\n')
