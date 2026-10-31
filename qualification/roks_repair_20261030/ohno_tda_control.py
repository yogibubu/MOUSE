"""Separate historical Ohno J/K control, lambda=.9, actual GFN2 MOs.

This is a different kernel from the variational SCC-response TDA benchmark.
No DFT orbitals, gaps or extra ground SCF enter this diagnostic.
"""
from remote_bootstrap import HOST,BASE
from pathlib import Path
import json,time,numpy as np
from spin_kernel_probe import state_from,write,key,digest,EV
from extent_support import source_molecule
from matrix_gfn2.imported_tda import ImportedTDAReference,ImportedTDASelection,ImportedTDAOperator,OhnoTDAParameters
from pyscf import lib
lib.num_threads(2);ROOT=BASE/'results/ohno_tda_control';ROOT.mkdir(exist_ok=True)
cases=json.loads((BASE/'results/full_reference_inputs/frozen_relaxation/plan.json').read_text())['cases']
for case in cases:
 name=case['name'];out=ROOT/name;out.mkdir(exist_ok=True);final=out/'result.json'
 request=key(dict(script=digest(__file__),source=digest(case['targets'][0]['checkpoint']),ground_exchange=.25,lambda_exchange=.9,extent_damping=.5))
 if final.exists():assert json.loads(final.read_text())['request_hash']==request;continue
 r,xyz=state_from(case['targets'][0]);s=r.reference_state;mol,t=source_molecule(s)
 moment_start=time.perf_counter();position=-np.array([t.T@v@t for v in mol.intor('int1e_r',comp=3)]);second=t.T@mol.intor('int1e_r2')@t;moment_seconds=time.perf_counter()-moment_start
 ref=ImportedTDAReference(s.model_id,s.atomic_numbers,s.coordinates_bohr,s.ao_atom,s.overlap,s.fock,
  s.coefficients,s.orbital_energies_hartree,s.alpha_occupations+s.beta_occupations,position,second)
 t0=time.perf_counter();op=ImportedTDAOperator(ref,parameters=OhnoTDAParameters.from_ground_state_exchange(.25,response_exchange_scale=.9,extent_exchange_damping=.5),selection=ImportedTDASelection(1000.))
 matrix=op.apply_block(np.eye(op.dimension));values,vectors=np.linalg.eigh(matrix);seconds=time.perf_counter()-t0
 ii=np.repeat(op.occupied,len(op.virtual));aa=np.tile(op.virtual,len(op.occupied));np.savez_compressed(out/'tda.npz',matrix=matrix,energies=values,vectors=vectors,occupied=ii,virtual=aa)
 rows=[];hold=(BASE/'holdout_data'/name/'case.json').exists()
 for target in case['targets']:
  anchor=np.zeros(len(ii));anchor[np.flatnonzero((ii==target['hole'])&(aa==target['particle']))[0]]=1
  if hold:
   with np.load(BASE/'holdout_data'/name/'reference_transition.npz') as f:anchor=f['amplitude'].ravel()
  score=abs(anchor@vectors);root=int(np.argmax(score));margin=float(score[root]-np.max(np.delete(score,root),initial=0.))
  rows.append(dict(state=target['id'],tbe_ev=target['tbe_ev'],energy_ev=float(values[root]*EV),error_ev=float(values[root]*EV-target['tbe_ev']),overlap=float(score[root]),margin=margin,qualified=bool(score[root]>=.7 and margin>=.05)))
 result=dict(host=HOST,request_hash=request,molecule=name,cohort='holdout' if hold else 'development',
  kernel='Ohno J/K; different from SCC variational response',orbitals='GFN2',ground_state_exchange_convention=.25,
  response_exchange_scale=.9,final_response_exchange=.225,extent_exchange_damping=.5,
  solver_seconds=seconds,moments_seconds=moment_seconds,rows=rows)
 write(final,result);print(result,flush=True)
