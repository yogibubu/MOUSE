"""Recover independently prepared ROKS roots after finite orbital perturbations."""
from pathlib import Path
import socket,runpy,json,time
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
BASE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101');HERE=BASE/'oracle/MOUSE-closing-validation-20261101';runpy.run_path(str(BASE/'source/qualification/remote_bootstrap.py'))
import numpy as np
from scipy.linalg import expm
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.nosi_surface import PreparedNOSIModel
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from matrix_gfn2._roks_matrix_free import solve
from threadpoolctl import threadpool_limits
rows=[]
with threadpool_limits(2):
 for cohort,name,krylov in [('closing_blind','Ethylene','gmres'),('additional_pi_eta060','Pyrrole','gmres'),('closing_blind','Ethylene','minres'),('additional_pi_eta060','Pyrrole','minres')]:
  model=PreparedNOSIModel(HERE/'disentangled_bath_spectral_block_reference_eta_0.600000'/cohort/name/'model.json');s=model.anchor;no=s.electron_count//2;n=len(s.overlap);calc=GFN2Calculator([{1:'H',6:'C',7:'N'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.);ctx=prepare_gfn2_deltascf(calc,s.coordinates_bohr/ANGSTROM_TO_BOHR,orbital_reference=s,spin_reference=model.response);ground=model.arrays['ground'];c=model.arrays['states'][0];mu=model.arrays['multipliers'][0];transition=np.outer(ground[:,:no].T@s.overlap@c[:,0],ground[:,no:].T@s.overlap@c[:,no]);transition/=np.linalg.norm(transition)
  occ=np.zeros((2,n));occ[:,:no]=1;occ[0,0]=0;occ[0,no]=1;triplet=np.array([np.maximum(*occ),np.minimum(*occ)])
  def energy(coefficient):return sum(weight*ctx.fock_and_energy(np.array([(coefficient*o)@coefficient.T for o in occupations]))[1] for weight,occupations in [(2.,occ),(-1.,triplet)])
  reference=energy(c);rng=np.random.default_rng(401);k=rng.normal(size=(n,n));k=k-k.T;k*=.006/np.linalg.norm(k);start=time.monotonic();coeff,lagrange,report=solve(ctx,c@expm(k),ground,transition,mu,tolerance=1e-11,krylov=krylov,local_state_following=krylov=='minres',residual_trust_region=True,maximum_step=.001);error=abs(report['energy_hartree']-reference);assert error<1e-9 and report['residual']<1e-11
  rows.append(dict(molecule=name,krylov=krylov,model=model.fingerprint,energy_error_hartree=error,wall_seconds=time.monotonic()-start,report=report));print(name,krylov,error,report['residual'],flush=True)
 (HERE/'matrix_free_roks_certificate.json').write_text(json.dumps(dict(host=socket.getfqdn(),rows=rows),indent=2)+'\n')
