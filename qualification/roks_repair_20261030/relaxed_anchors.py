"""Export fixed local ROKS potentials anchored to qualified relaxed DFT energies.

Only the vertical energy at R0 is matched to DFT. The gradient and full analytic
nuclear Hessian belong to the frozen-response semiempirical potential.
"""
from remote_bootstrap import HOST,BASE
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
import json,time,traceback,numpy as np
from spin_kernel_probe import state_from,write,key,digest,EV
from matrix_gfn2 import (GFN2Calculator,FrozenResponseReference,FrozenROKSAnchor,AnchoredROKSResult,
 evaluate_anchored_roks,evaluate_anchored_roks_analytic_hessian,save_frozen_roks_anchor,
 load_frozen_roks_anchor,save_anchored_roks_point,load_anchored_roks_point)
from matrix_gfn2.deltascf_derivatives import load_derivative_point
from matrix_link.scan import QMScanBackend
from pyscf import lib
lib.num_threads(2)
ROOT=BASE/'results/relaxed_anchors';ROOT.mkdir(exist_ok=True)
PLAN=json.loads((BASE/'results/full_reference_inputs/frozen_relaxation/plan.json').read_text())['cases']

def run(case,target):
 name=case['name'];sid=target['id'];full=BASE/'results/full_reference'/name/sid/'result.json'
 if not full.exists():return
 dft=json.loads(full.read_text());assert dft['qualified']
 prefix=BASE/('results/holdout' if (BASE/'holdout_data'/name/'case.json').exists() else 'results')
 model=prefix/'response_kernel'/name
 row=next(r for r in json.loads((model/'result.json').read_text())['records'] if r['id']==sid);assert row['qualified']
 r0,xyz=state_from(target);s=r0.reference_state
 with np.load(model/'kernel.npz') as f:
  ref=FrozenResponseReference(f['atomic_numbers'],f['orbital_shell'],f['spin_coupling'],
   'PBE0/frozen-charge-spin-response/'+str(f['request_hash']),.9,f['charge_correction'],f['reference_shell_populations'])
 r=load_derivative_point(model/sid/'converged.npz',request_hash=row['request_hash'],coordinates_angstrom=xyz,require_gradient=False)
 assert ref.fingerprint==r.spin_reference_fingerprint
 anchor=FrozenROKSAnchor(ref,r.occupations,s.coordinates_bohr,dft['excitation_ev']/EV,row['raw_roks_ev']/EV,
  'relaxed BHANDHLYP/aug-cc-pVDZ energy '+digest(full)+'; frozen PBE0 response '+digest(model/'kernel.npz'),kind='relaxed_vertical')
 out=ROOT/name/sid;out.mkdir(parents=True,exist_ok=True);path=out/'model.npz'
 if path.exists():assert load_frozen_roks_anchor(path).fingerprint==anchor.fingerprint
 else:save_frozen_roks_anchor(path,anchor)
 request=key(dict(anchor=anchor.fingerprint,source=digest(model/sid/'converged.npz')))
 calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
 ground=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=r0.ground_energy_hartree)
 point=out/'central.npz'
 if point.exists():central=load_anchored_roks_point(point,anchor,request_hash=request,coordinates_angstrom=xyz)
 else:
  central=evaluate_anchored_roks(calc,xyz,anchor,reference_result=AnchoredROKSResult(r,anchor),ground_state=ground,
   gradient=True,orbital_solver='gmf',saddle_index=r.orbital_saddle_index,orbital_tolerance=1e-10,maximum_iterations=100)
  save_anchored_roks_point(point,central,request_hash=request,coordinates_angstrom=xyz)
 pathh=out/'analytic.npz'
 if pathh.exists():
  with np.load(pathh) as f:
   assert str(f['anchor'])==anchor.fingerprint
   hessian=f['hessian'];curvatures=f['curvatures'];residual=float(f['response_error']);seconds=float(f['seconds'])
 else:
  start=time.perf_counter();h=evaluate_anchored_roks_analytic_hessian(calc,xyz,anchor,reference_result=central)
  hessian=h.hessian_hartree_per_bohr2;curvatures=np.linalg.eigvalsh(h.orbital_hessian_hartree)
  residual=h.maximum_response_residual;seconds=time.perf_counter()-start
  np.savez_compressed(pathh,anchor=anchor.fingerprint,hessian=hessian,gradient=central.gradient_hartree_per_bohr,
   curvatures=curvatures,response_error=residual,seconds=seconds)
 backend=QMScanBackend(name='matrix_gfn2',electronic_state=1,native_excited_method='roks',
  deltascf_promotion=(target['hole'],target['particle']),orbital_saddle_index=central.orbital_saddle_index,
  native_roks_anchor_path=out/'model.npz')
 write(out/'backend.json',asdict(backend))
 terr=float(np.max(abs(hessian@np.kron(np.ones((len(xyz),1)),np.eye(3)))))
 index=int(np.count_nonzero(curvatures < -1e-7));energy_error=abs(central.excitation_energy_hartree*EV-dft['excitation_ev'])
 report=dict(host=HOST,molecule=name,state=sid,qualified=True,request_hash=request,anchor_fingerprint=anchor.fingerprint,
  anchor_kind=anchor.kind,anchor_file_sha256=digest(out/'model.npz'),dft_reference_sha256=digest(full),
  excitation_ev=central.excitation_energy_hartree*EV,tbe_ev=target['tbe_ev'],
  anchor_reproduction_error_ev=energy_error,gradient_available=True,hessian_dimension=xyz.size,
  analytic_hessian_seconds=seconds,nuclear_displacements_in_analytic_method=0,
  symmetry_error=float(np.max(abs(hessian-hessian.T))),translation_sum_rule=terr,
  orbital_response_residual=residual,gmf_index=central.orbital_saddle_index,hessian_orbital_index=index,
  scope='fixed local semiempirical potential; energy matches relaxed DFT at R0; derivatives are not DFT derivatives',
  blind_accuracy_validation=False,default_changed=False)
 report['passed']=bool(energy_error<1e-6 and terr<1e-7 and residual<1e-8 and index==central.orbital_saddle_index)
 write(out/'validation.json',report);print('ANCHOR',report,flush=True);assert report['passed']

if __name__=='__main__':
 for case in PLAN:
  for target in case['targets']:
   try:run(case,target)
   except Exception as exc:
    path=ROOT/case['name']/target['id'];path.mkdir(parents=True,exist_ok=True)
    write(path/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()));print(traceback.format_exc(),flush=True)
