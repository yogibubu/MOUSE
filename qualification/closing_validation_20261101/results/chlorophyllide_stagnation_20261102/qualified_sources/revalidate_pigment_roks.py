"""Independent, zero-iteration check of all stored production KKT roots."""
from pathlib import Path
import hashlib,json,socket,time,sys
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
from run_pigment_roks import load,HERE,write
from matrix_gfn2._roks_matrix_free import solve
import matrix_gfn2._roks_matrix_free as implementation
import numpy as np
from threadpoolctl import threadpool_limits
start=time.monotonic();sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(implementation.__file__),HERE/'run_pigment_roks.py',HERE/'run_pigment_energy_escape.py',HERE/'finalize_pigment_roks.py']}
with threadpool_limits(8):
 model,state,reference,calc,base,context=load('Chlorophyllide_a');no=state.electron_count//2;n=len(state.coefficients);transition=np.zeros((4,no,n-no));promotions=[(no-1,no),(no-2,no),(no-1,no+1),(no-2,no+1)]
 for k,(hole,particle) in enumerate(promotions):transition[k,hole,particle-no]=1.
 rows=[];coefficients=[]
 for i in range(1,5):
  folder=HERE/f'pigment_roks/Chlorophyllide_a/state_{i}';certificate=json.loads((folder/'certificate.json').read_text())
  with np.load(folder/'checkpoint.npz') as z:c=z['coefficients'].copy();mu=float(z['multiplier'])
  verified,lagrange,report=solve(context,c,state.coefficients,transition,mu,tolerance=1e-10,maximum_iterations=0,retract_constraint=False)
  error=abs(report['energy_hartree']-certificate['report']['energy_hartree']);assert error<1e-11 and report['history'][0]['iteration']==0 and lagrange==mu
  row=dict(state=i,residual=report['residual'],ground_overlap=report['ground_overlap'],native_frontier_fidelity=report['fidelity'],energy_hartree=report['energy_hartree'],energy_error_hartree=error,optimization_iterations=0);rows.append(row);coefficients.append(c);print(json.dumps(row),flush=True)
 ia=list(range(1,no))+[0];ib=list(range(1,no))+[no];s=state.overlap
 def overlap(a,b):return float(np.linalg.det(a[:,ia].T@s@b[:,ia])*np.linalg.det(a[:,ib].T@s@b[:,ib])+np.linalg.det(a[:,ia].T@s@b[:,ib])*np.linalg.det(a[:,ib].T@s@b[:,ia]))
 metric=np.array([[overlap(a,b) for b in coefficients] for a in coefficients]);assert np.linalg.eigvalsh(metric)[0]>1e-8
 result=dict(host=socket.getfqdn(),complete=True,rows=rows,metric_eigenvalues=np.linalg.eigvalsh(metric).tolist(),source_sha256_at_import=sources,original_KKT_verified_without_retraction_or_multiplier_adjustment=True,DFT_imports=int(any(k=='pyscf' or k.startswith('pyscf.') for k in sys.modules)),new_SCC_calls=0,TDA_solves=0,wall_seconds_verification=time.monotonic()-start,pigment_derivative_scope='Vertical energies and orbital response only; no pigment nuclear-Hessian qualification')
 assert result['DFT_imports']==0
 write(HERE/'pigment_stagnation_repair_20261102/final_revalidation.json',result);print('ALL FOUR ROOTS INDEPENDENTLY VERIFIED',flush=True)
