"""Oracle-only cold/resumed/displaced portable multi-CSF LINK qualification."""
from pathlib import Path
import os,sys,socket,fcntl,time,json,hashlib,shutil
assert socket.getfqdn()=='oracle.sns.it'
V=Path('/home/enzo/MOUSE-orbital-performance-20261031');W=V/'multicsf_integration'
os.environ.update(MOUSE_VALIDATION_ROOT=str(W),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2');sys.path.insert(0,str(W/'source/qualification'));import remote_bootstrap
import numpy as np
from types import SimpleNamespace
from dataclasses import replace
from threadpoolctl import threadpool_limits
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from matrix_gfn2 import GFN2Calculator,MultiCSFModel
from matrix_link.scan import QMScanBackend,ScanPoint,_run_matrix_gfn2_scan_point
from matrix_link.qmqm_component import CappedQMComponent
from matrix_link.optimizer import GeometryEvaluationService,_restart_identity_value

with threadpool_limits(limits=2):
 lock=(V.parent/'MOUSE-publication-validation-20261030/blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 modelpath=W/'models/Furan/B022/model.json';model=MultiCSFModel(modelpath);symbols=[{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in model.anchor.atomic_numbers]
 def session():
  calc=GFN2Calculator(symbols,temperature_kelvin=0.)
  calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('Unexpected SCC/DFT/TDA call'))
  return SimpleNamespace(_calculator=calc)
 spec=QMScanBackend(name='matrix_gfn2',electronic_state=1,native_excited_method='multicsf_50_50',native_multicsf_model_path=modelpath,properties=('energy','gradient','hessian'))
 root=W/'link_validation';root.mkdir(exist_ok=True);run=root/'accepted_run';started=time.monotonic();sess=session()
 result=_run_matrix_gfn2_scan_point(spec,tuple(symbols),ScanPoint(0,0.,model.xyz),sess,run_dir=run);cold=time.monotonic()-started
 with np.load(V/'pi_pi_repair/nuclear_validation/Furan_excited_total_analytic.npz') as a:
  energy_error=abs(result.energy_hartree-float(a['energy']));gradient_error=float(np.max(abs(result.gradient_hartree_per_bohr-a['gradient'].ravel())));hessian_error=float(np.max(abs(result.hessian_hartree_per_bohr2-a['hessian'])))
 assert energy_error<1e-10 and gradient_error<1e-9 and hessian_error<1e-9
 manifest=run/'multicsf_accepted/accepted.json';original=manifest.read_bytes();component=sess._link_roks_component;retained=component.reference
 x=model.xyz.copy();x.ravel()[8]+=.004/ANGSTROM_TO_BOHR
 trial=component.evaluate(x,compute_hessian=False)
 assert component.reference is retained and manifest.read_bytes()==original
 assert min(trial.execution['component_state_overlaps'])>.9
 # Direct geometry-optimizer dispatch uses the same component and analytic route.
 service=object.__new__(GeometryEvaluationService);service.backend=spec;service._orbital_component=component
 dispatched=service._evaluate_builtin_backend_with_state_following(ScanPoint(1,.004,x),tag='trial',requested_properties=('energy','gradient'))
 assert abs(dispatched.energy_hartree-trial.energy_hartree)<1e-10
 # A fresh resident session restores the accepted point, never a rejected trial.
 start=time.monotonic();resumed=CappedQMComponent(spec,session(),run);warm=resumed.evaluate(model.xyz,compute_hessian=True);resume_seconds=time.monotonic()-start
 assert abs(warm.energy_hartree-result.energy_hartree)<1e-12 and np.max(abs(warm.hessian_hartree_per_bohr2-result.hessian_hartree_per_bohr2))<1e-12
 assert np.max(abs(resumed.reference[0].reference_state.coordinates_bohr-model.anchor.coordinates_bohr))<1e-12
 # Relative bundle paths support relocating every preparation artifact together.
 portable=root/'relocated_model'
 if not portable.exists():shutil.copytree(modelpath.parent,portable)
 relocated=CappedQMComponent(replace(spec,native_multicsf_model_path=portable/'model.json'),session(),run)
 assert relocated.reference[0].model_fingerprint==model.fingerprint
 # Tampered accepted point is rejected before any state is advanced.
 row=json.loads(manifest.read_text());record=run/'multicsf_accepted'/row['path'];metadata=json.loads(record.read_text());point=record.parent/'point.npz';raw=point.read_bytes()
 try:
  point.write_bytes(raw+b'changed')
  try:CappedQMComponent(spec,session(),run)
  except ValueError:pass
  else:raise AssertionError('corrupt accepted state was restored')
 finally:point.write_bytes(raw)
 assert manifest.read_bytes()==original
 # Model bytes are checked before optimizer cache access too.
 arrays=modelpath.parent/'arrays.npz';raw=arrays.read_bytes()
 try:
  arrays.write_bytes(raw+b'changed')
  try:service.evaluate(np.zeros(model.xyz.size),tag='cached')
  except ValueError:pass
  else:raise AssertionError('changed model escaped cache validation')
 finally:arrays.write_bytes(raw)
 report=dict(host=socket.getfqdn(),model_fingerprint=model.fingerprint,rank=model.rank,cold_seconds=cold,resume_seconds=resume_seconds,energy_error=energy_error,gradient_max_error=gradient_error,hessian_max_error=hessian_error,displaced_parent_overlaps=trial.execution['component_state_overlaps'],rejected_trial_preserves_accepted_state=True,portable_bundle_restart=True,tamper_detection=True,direct_optimizer_dispatch=True,ground_scc_calls=0,tda_calls=0,analytic_hessian=True,source_runtime=str(W/'source'))
 (root/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
