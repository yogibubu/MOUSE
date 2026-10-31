"""Separate XC nonlinearity, native higher orders, and the prescribed lambda.

All values are diagnostics on quadratic-model orbitals, not new predictions.
The nonlinear completion includes +(.25-.225)/4 [dP K dP + M K M], which
preserves the PBE0 first derivative at the anchor and makes its Hessian use
exactly the same lambda=.9. No fitted state offset is added.
"""
from remote_bootstrap import HOST,BASE
from pathlib import Path
from types import SimpleNamespace
import sys,json,numpy as np
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import plan,state_from,write,EV
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.deltascf_derivatives import load_derivative_point
from pyscf import scf,ao2mo,lib
lib.num_threads(4)
for case in [c for c in plan() if c['name'] in ['nitroxyl','uracil']]:
 out=BASE/'results/full_response'/case['name'];final=out/'remainder_decomposition.json'
 if final.exists():continue
 r0,xyz=state_from(case['targets'][0]);s=r0.reference_state;n=len(s.overlap)
 calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
 native=prepare_gfn2_deltascf(calc,xyz,ground_state=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=r0.ground_energy_hartree))
 checkpoint=out/'partial_quadrature.npz'
 if checkpoint.exists():
  with np.load(checkpoint) as f:eri=f['eri']
 else:
  with np.load(Path('/home/enzo/MOUSE-roks-repair-20261030/results/spin_kernel')/case['name']/'kernel.npz') as f:lift=f['embedding']
  mol=scf.chkfile.load_mol(str(Path(case['reference'])/'backend.chk'))
  eri=ao2mo.incore.general(mol.intor('int2e',aosym='s8'),(lift,lift,lift,lift),compact=False).reshape(n,n,n,n)
 p0=(s.coefficients*(s.alpha_occupations+s.beta_occupations))@s.coefficients.T
 f0,e0=native.fock_and_energy(np.array([.5*p0,.5*p0]))
 def remainder(p):
  dp=p-p0;_,e=native.fock_and_energy(np.array([.5*p,.5*p]))
  return e-e0-np.sum(f0[0]*dp)-.5*np.sum(dp*native.kernel(dp))
 def hf_completion(ps):
  dp=ps.sum(axis=0)-p0;m=ps[0]-ps[1]
  kdp=np.einsum('ikjl,kl->ij',eri,dp,optimize=True);km=np.einsum('ikjl,kl->ij',eri,m,optimize=True)
  return .025/4*float(np.sum(dp*kdp)+np.sum(m*km))
 with np.load(out/'tda.npz') as f:cg=f['ground_coefficients']
 pg=np.array([(c*o)@c.T for c,o in zip(cg,[s.alpha_occupations,s.beta_occupations])]);hg=hf_completion(pg);rg=remainder(pg.sum(axis=0))
 previous=json.loads((out/'nonlinear_diagnostic.json').read_text());rows=[]
 for target in case['targets']:
  row=json.loads((out/target['id']/'result.json').read_text())
  if not row.get('qualified'):continue
  r=load_derivative_point(row['checkpoint'],request_hash=row['request_hash'],coordinates_angstrom=xyz,require_gradient=False);c=r.common_coefficients
  pm=np.array([(c*o)@c.T for o in r.occupations]);pt=np.array([(c*o)@c.T for o in r.triplet_occupations])
  hc=(2*hf_completion(pm)-hf_completion(pt)-hg)*EV;rn=(remainder(pm.sum(axis=0))-rg)*EV
  prev=next(v for v in previous['rows'] if v['state']==target['id']);complete=prev['exact_embedded_dft_on_same_orbitals_ev']+hc
  difference=complete-row['excitation_ev'];xc=difference+rn
  rows.append(dict(state=target['id'],quadratic_model_ev=row['excitation_ev'],nonlinear_lambda_consistent_completion_ev=complete,
   hf_response_scaling_completion_ev=hc,completion_minus_model_ev=difference,
   native_higher_order_contribution_to_gap_ev=rn,xc_higher_order_contribution_to_gap_ev=xc,
   tbe_ev=target['tbe_ev'],stationary_nonlinear_solution=False))
 report=dict(host=HOST,molecule=case['name'],rows=rows,lambda_exchange=.9,scope='functional remainder diagnostic, not an energy-offset correction or a stationary prediction')
 write(final,report);print(report,flush=True)
