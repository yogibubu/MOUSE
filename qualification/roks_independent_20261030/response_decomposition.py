"""Exact additive energy accounting on ablation stationary orbitals.

This separates native energy on the model orbitals, the reference Fock term,
charge response, and spin replacement. It never changes a predicted energy.
Ground states are reused across models with identical charge/Fock terms.
"""
from remote_bootstrap import HOST,BASE
from pathlib import Path
from types import SimpleNamespace
import sys,json,numpy as np
from threadpoolctl import threadpool_limits
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import plan,state_from,write,digest,key,EV
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.deltascf import prepare_gfn2_deltascf,run_deltascf
from matrix_gfn2.deltascf_derivatives import load_derivative_point
from full_response import VerticalReference
from pyscf import lib
lib.num_threads(2)
ROOT=BASE/'results/response_ablation'
summary=json.loads((ROOT/'summary.json').read_text())
cases={c['name']:c for c in plan() if c['name'] in ['nitroxyl','uracil']}
cases.update({n:json.loads((BASE/'prepared_B'/n/'case.json').read_text()) for n in ['acetaldehyde','acetone','formamide']})
rows=[]
with threadpool_limits(limits=2,user_api='blas'):
 for name,case in cases.items():
  old,xyz=state_from(case['targets'][0]);s=old.reference_state
  ground=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=old.ground_energy_hartree)
  calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
  native=prepare_gfn2_deltascf(calc,xyz,ground_state=ground)
  for label in summary['statistics']:
   folder=(BASE/'results/full_response' if label=='full' else ROOT/label)/name
   with np.load(folder/'model.npz') as f:data={k:f[k].copy() for k in f.files};request=str(f['request_hash'])
   ref=VerticalReference(s,native.kernel.partition,data,request)
   ctx=prepare_gfn2_deltascf(calc,xyz,ground_state=ground,spin_reference=ref)
   gdir=ROOT/'ground_states'/name;gdir.mkdir(parents=True,exist_ok=True)
   gh=key(dict(linear=data['linear'].tolist(),charge=data['charge'].tolist(),q0=data['q0'].tolist(),source=digest(BASE/'results/full_response'/name/'model.npz')))
   gpfile=gdir/(gh+'.npz')
   if gpfile.exists():
    with np.load(gpfile) as f:cg=f['coefficients'];ge=float(f['energy_hartree'])
   else:
    gp=run_deltascf(s.overlap,np.array([s.coefficients]*2),np.array([s.alpha_occupations,s.beta_occupations]),ctx.fock_and_energy,selection='aufbau',density_tolerance=1e-10,orbital_tolerance=1e-10,energy_tolerance=1e-12,maximum_iterations=500)
    assert gp.converged and gp.orbital_residual<=1e-10;cg=gp.coefficients;ge=gp.energy_hartree
    np.savez_compressed(gpfile,coefficients=cg,energy_hartree=ge,orbital_residual=gp.orbital_residual)
   pg=np.array([(c*o)@c.T for c,o in zip(cg,[s.alpha_occupations,s.beta_occupations])])
   b=.5*(s.overlap@data['operators']+data['operators']@s.overlap)
   def terms(ps):
    q=np.einsum('aij,ij->a',b,ps.sum(axis=0));dq=q-data['q0'];m=np.einsum('aij,ij->a',b,ps[0]-ps[1])
    ns=native.polarization.evaluate(ps)[1]
    return np.array([native.fock_and_energy(ps)[1],data['linear']@q,.5*dq@data['charge']@dq,.5*m@data['spin']@m-ns])
   tg=terms(pg);assert abs(tg.sum()-ge)<1e-9
   for target in case['targets']:
    result=json.loads((folder/target['id']/'result.json').read_text())
    if not result.get('qualified'):continue
    r=load_derivative_point(result['checkpoint'],request_hash=result['request_hash'],coordinates_angstrom=xyz,require_gradient=False)
    c=r.common_coefficients;pm=np.array([(c*o)@c.T for o in r.occupations]);pt=np.array([(c*o)@c.T for o in r.triplet_occupations])
    te=2*terms(pm)-terms(pt);contribution=(te-tg)*EV
    closure=float(contribution.sum()-result['excitation_ev']);assert abs(closure)<1e-7
    rows.append(dict(molecule=name,state=target['id'],variant=label,excitation_ev=result['excitation_ev'],error_ev=result['error_ev'],
     native_on_model_orbitals_ev=float(contribution[0]),reference_fock_ev=float(contribution[1]),charge_response_ev=float(contribution[2]),spin_replacement_ev=float(contribution[3]),closure_error_ev=closure,
     ground_energy_error_hartree=float(ge-r.ground_energy_hartree),ground_checkpoint=str(gpfile)))
 write(ROOT/'decomposition.json',dict(host=HOST,script_sha256=digest(__file__),rows=rows,passed=True,scope='exact accounting on stationary model orbitals; no new predictions or offsets'))
print('DECOMPOSITION',len(rows),'max closure',max(abs(r['closure_error_ev']) for r in rows),flush=True)
