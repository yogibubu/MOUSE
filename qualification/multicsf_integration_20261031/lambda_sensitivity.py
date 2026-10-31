"""Exchange-response sensitivity with frozen one-body DFT anchor and rank.
Only the diagonal exact-exchange RESPONSE changes. Bare offdiagonal Slater-
Condon couplings, semilocal response and equal parent weights are unchanged.
No fitting and no new DFT calculation. This is a declared diagnostic variant.
"""
from pathlib import Path
import os,sys,socket,fcntl,json,time,copy
assert socket.getfqdn()=='oracle.sns.it'
V=Path('/home/enzo/MOUSE-orbital-performance-20261031');W=V/'multicsf_integration'
os.environ.update(MOUSE_VALIDATION_ROOT=str(W),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2');sys.path.insert(0,str(W/'source/qualification'));import remote_bootstrap
import numpy as np
from dataclasses import replace
from threadpoolctl import threadpool_limits
from matrix_gfn2 import GFN2Calculator,MultiCSFModel
from matrix_gfn2.multicsf_frame import RelativeActiveFrame
from matrix_gfn2.restricted_ground import evaluate_restricted_ground
EV=27.211386245988
with threadpool_limits(limits=2):
 lock=(V.parent/'MOUSE-publication-validation-20261030/blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 out=W/'lambda_sensitivity';out.mkdir(exist_ok=True);rows=[]
 for name,sid in [('Cyclopentadiene','B016'),('Furan','B022'),('Phthalimide','B024')]:
  base=MultiCSFModel(W/f'models/{name}/{sid}/model.json');n=len(base.anchor.overlap);ops=base.responses[0].operators;flat=ops.reshape(len(ops),-1)
  assert np.max(abs(flat@flat.T-np.eye(len(flat))))<1e-10
  exchange=flat@base.eri.eri.transpose(0,2,1,3).reshape(n*n,n*n)@flat.T;exchange=.5*(exchange+exchange.T)
  calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in base.anchor.atomic_numbers],temperature_kelvin=0.)
  for lam in [.85,.9,.95]:
   path=out/f'{sid}_{lam:.2f}.json'
   if path.exists():rows.append(json.loads(path.read_text()));continue
   start=time.monotonic();model=copy.copy(base);delta=.25*(lam-.9)
   model.responses=tuple(replace(r,response_exchange_scale=lam,pair_charge_hartree=r.pair_charge_hartree-.5*delta*exchange,pair_spin_hartree=r.pair_spin_hartree-.5*delta*exchange,source_model=r.source_model+f':lambda_response_sensitivity:{lam}') for r in base.responses) if lam!=.9 else base.responses
   objects,point=model.objectives(calc,base.xyz);frame=RelativeActiveFrame(objects);point=frame.polish(point);ee,g,h,*_=frame.evaluate(point);curvature=np.linalg.eigvalsh(h);fidelity=[o.fidelity(c)[0] for o,c in zip(objects,point)];assert min(fidelity)**2>.5 and np.min(abs(curvature))>1e-7
   grounds=[]
   for seed,ref in zip(base.grounds[1:],model.responses):
    if lam==.9:grounds.append(seed);continue
    # The relabelled, explicitly NONCONVERGED object supplies orbitals only.
    guess=replace(seed.reference,spin_reference_fingerprint=ref.fingerprint,converged=False,orbital_residual=np.inf)
    grounds.append(evaluate_restricted_ground(calc,base.xyz,template=base.anchor,spin_reference=ref,reference_result=guess,gradient=False))
   energy=(ee-.5*sum(r.energy_hartree for r in grounds))*EV
   row=dict(host=socket.getfqdn(),molecule=name,state=sid,lambda_exchange=lam,excitation_ev=energy,rank=base.rank,weights=[.5,.5],energy_offset_ev=0.,residual=float(np.max(abs(g))),minimum_absolute_curvature=float(np.min(abs(curvature))),fidelity=fidelity,seconds=time.monotonic()-start,model_fingerprint=base.fingerprint,scope='diagonal exact-exchange response only; one-body DFT anchor and bare CSF couplings fixed; no refit')
   path.write_text(json.dumps(row,indent=2)+'\n');rows.append(row);print(name,lam,energy,flush=True)
 (out/'summary.json').write_text(json.dumps(dict(host=socket.getfqdn(),rows=rows,production_lambda=.9,parameter_selection_from_results=False),indent=2)+'\n')
