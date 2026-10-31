"""Migrate the completed planar local-frame phase into a stationary-root run.
All five electronic checkpoints and the exact nuclear Hessian are retained.
"""
import socket,json,sys,importlib.util,shutil,hashlib,os
from pathlib import Path
import numpy as np
assert socket.getfqdn()=='oracle.sns.it'
BASE=Path('/home/enzo/MOUSE-roks-local-newton-20261030');PARENT=Path('/home/enzo/MOUSE-roks-local-20261030')
protocol=json.loads((PARENT/'protocol.json').read_text());protocol.update(schema='mouse.roks.local_covalent_frames.newton.v1',tracking='full local CSF and local-frame original active-pair >=0.7; Newton continues the stationary root and certifies its observed analytic index without prescribing an index ceiling',restart='completed planar formamide states, native/corrected S0 orbitals and exact nuclear Hessian migrated; no repeated completed optimization')
(BASE/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
spec=importlib.util.spec_from_file_location('campaign',BASE/'source/qualification/run_newton_branch.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from matrix_gfn2 import GroundReferencedROKSResult,combine_roks_results
from matrix_gfn2.restricted_ground import evaluate_restricted_ground
from matrix_link.composite_roks_component import persist_accepted
from matrix_link.qmqm_component import CappedQMComponent
from matrix_link.scan import QMScanBackend
from types import SimpleNamespace
with m.threadpool_limits(limits=2):
 source=PARENT/'results/formamide';root=BASE/'results/formamide';root.mkdir(parents=True,exist_ok=True)
 if (root/'optimization/roks_50_50_accepted/accepted.json').exists():raise SystemExit(0)
 with np.load(source/'analytic_stationary_0.npz') as d:xyz=d['xyz'].copy();h=d['hessian'].copy()
 found=[]
 for path in (source/'optimization/roks_50_50_accepted').glob('*/reference_0.npz'):
  with np.load(path) as d:
   if np.max(abs(d['coordinates_angstrom']-xyz))<1e-12:found.append(path.parent)
 if len(found)!=1:raise ValueError('ambiguous planar accepted generation: '+str(found))
 generation=found[0];data=json.loads((source/'prepared.json').read_text());data['protocol_sha256']=m.sha(BASE/'protocol.json')
 m.write(root/'prepared.json',data)
 for i in range(2):shutil.copy2(source/f'anchor_{i}.npz',root/f'anchor_{i}.npz')
 def retained(path):
  with np.load(path) as d:request=str(d['request_hash'])
  return m.load(path,request)
 excited=[retained(generation/f'reference_{i}.npz') for i in range(2)]
 for i,r in enumerate(excited):m.save_derivative_point(root/f'seed_{i}.npz',r,request_hash='local_seed_v1',coordinates_angstrom=xyz)
 raw_grounds=[retained(generation/f'ground_{i}.npz') for i in range(3)]
 refs=[m.load_response_reference(path) for path in data['response_paths']];calc=m.GFN2Calculator(data['atoms'],temperature_kelvin=0.)
 calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('SCC invoked'))
 grounds=[evaluate_restricted_ground(calc,xyz,template=r.reference_state,reference_result=r,spin_reference=ref,gradient=True) for r,ref in zip(raw_grounds,[None]+refs)]
 if any(r.iterations!=0 for r in grounds):raise ValueError('retained S0 orbitals were not stationary')
 result=GroundReferencedROKSResult(combine_roks_results(*excited,component_tracking='active_pair'),tuple(grounds))
 oldenergy=.5*sum(r.energy_hartree for r in excited)+raw_grounds[0].energy_hartree-.5*sum(r.energy_hartree for r in raw_grounds[1:])
 assert abs(result.energy_hartree-oldenergy)<1e-11
 backend=m.QMScanBackend(name='matrix_gfn2',electronic_state=1,native_excited_method='roks_50_50',deltascf_promotion=tuple(data['promotion']),native_roks_composite_response_paths=tuple(data['response_paths']),native_roks_composite_saddle_indices=tuple(data['saddle_indices']),native_roks_composite_reference_policy='geometry_only',native_roks_composite_component_tracking='active_pair',native_roks_composite_following='character_newton_index',native_roks_composite_ground_surface='native_plus_gaps')
 comp=CappedQMComponent(backend,SimpleNamespace(_calculator=calc),root/'optimization')
 anchors=[m.load(root/f'anchor_{i}.npz','local_anchor_v1') for i in range(2)]
 comp.anchor=(combine_roks_results(*anchors,component_tracking='active_pair'),tuple(r.to_apoc(1) for r in anchors))
 pair=(result,tuple(r.to_apoc(1) for r in excited));persist_accepted(comp,pair)
 shutil.copy2(source/'analytic_stationary_0.npz',root/'analytic_stationary_0.npz')
 values,vectors=np.linalg.eigh(h);seed=(vectors*np.maximum(abs(values),1e-4))@vectors.T
 np.savez_compressed(root/'optimizer_state.npz',coordinates_angstrom=xyz,hessian_seed=seed,escapes=0)
 m.write(root/'migration.json',dict(source_generation=str(generation),source_hessian=str(source/'analytic_stationary_0.npz'),source_hessian_sha256=m.sha(source/'analytic_stationary_0.npz'),ground_optimization_iterations=[r.iterations for r in grounds],energy_error=result.energy_hartree-oldenergy,new_policy='character_newton_index',source='same Hamiltonian, different stationary-root following policy'))
 print('MIGRATED',str(generation),flush=True)
