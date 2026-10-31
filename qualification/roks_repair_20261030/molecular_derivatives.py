"""Independent nuclear-gradient/Hessian validation of the frozen response surface."""
from remote_bootstrap import HOST,BASE
from response_kernel_probe import reference,plan,state_from,OLD,write,key,digest,EV
from pathlib import Path
import numpy as np,json,time
from types import SimpleNamespace
from matrix_gfn2 import (GFN2Calculator,FrozenROKSAnchor,AnchoredROKSResult,
 evaluate_anchored_roks,evaluate_anchored_roks_analytic_hessian,
 save_anchored_roks_point,load_anchored_roks_point,evaluate_gfn2_deltascf_hessian)
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.deltascf_derivatives import load_derivative_point
from pyscf import lib
lib.num_threads(2);ROOT=BASE/'results/derivatives/nitroxyl';ROOT.mkdir(parents=True,exist_ok=True)
case=next(c for c in plan() if c['name']=='nitroxyl');target=case['targets'][0]
r0,xyz=state_from(target);s=r0.reference_state
calc=GFN2Calculator([{1:'H',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
ground=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=r0.ground_energy_hartree)
ctx=prepare_gfn2_deltascf(calc,xyz,ground_state=ground);ref=reference(case,ctx)
folder=BASE/'results/response_kernel/nitroxyl';row=json.loads((folder/'result.json').read_text())['records'][0]
assert row['qualified']
result=load_derivative_point(folder/target['id']/'converged.npz',request_hash=row['request_hash'],coordinates_angstrom=xyz,require_gradient=False)
prior=json.loads((OLD/'frozen_relaxation/nitroxyl/result.json').read_text())['records'][0]
anchor=FrozenROKSAnchor(ref,result.occupations,s.coordinates_bohr,prior['dft_frozen_ev']/EV,row['frozen_ev']/EV,
 'PBE0 frozen excitation '+digest(OLD/'frozen_relaxation/nitroxyl/n_pi_dft.json'))
options=dict(orbital_solver='gmf',saddle_index=result.orbital_saddle_index,orbital_tolerance=1e-10,maximum_iterations=100)
point=ROOT/'central.npz';request=key(dict(anchor=anchor.fingerprint,tolerance=1e-10))
if point.exists():central=load_anchored_roks_point(point,anchor,request_hash=request,coordinates_angstrom=xyz)
else:
 central=evaluate_anchored_roks(calc,xyz,anchor,reference_result=AnchoredROKSResult(result,anchor),ground_state=ground,gradient=True,**options)
 save_anchored_roks_point(point,central,request_hash=request,coordinates_angstrom=xyz)
hpath=ROOT/'analytic.npz'
if hpath.exists():
 with np.load(hpath) as f:assert str(f['anchor'])==anchor.fingerprint;analytic=f['hessian'];response_error=float(f['response_error'])
else:
 start=time.perf_counter();h=evaluate_anchored_roks_analytic_hessian(calc,xyz,anchor,reference_result=central)
 analytic=h.hessian_hartree_per_bohr2;response_error=h.maximum_response_residual
 np.savez_compressed(hpath,anchor=anchor.fingerprint,hessian=analytic,gradient=central.gradient_hartree_per_bohr,
  frozen=h.frozen_orbital_hessian_hartree_per_bohr2,response=h.response_hessian_hartree_per_bohr2,
  response_error=response_error,seconds=time.perf_counter()-start)
numeric=evaluate_gfn2_deltascf_hessian(calc,xyz,*anchor.occupations,mode='roks',step_bohr=.001,
 reference_result=central.reference,spin_reference=ref,checkpoint_dir=ROOT/'numerical',**options)
np.savez_compressed(ROOT/'numerical.npz',hessian=numeric.hessian_hartree_per_bohr2,raw=numeric.raw_hessian_hartree_per_bohr2)
cache=next((ROOT/'numerical').iterdir());energy_gradient=[]
for coordinate in range(xyz.size):
 values=[]
 for sign in [-1,1]:
  with np.load(cache/f'{coordinate:04d}_{sign:+d}.npz') as f:values.append(float(f['result_energy_hartree']))
 energy_gradient.append((values[1]-values[0])/(2*.001))
gerr=float(np.max(abs(np.array(energy_gradient)-central.gradient_hartree_per_bohr.ravel())))
herr=float(np.max(abs(analytic-numeric.raw_hessian_hartree_per_bohr2)))
translation=np.kron(np.ones((len(xyz),1)),np.eye(3))
terr=float(np.max(abs(analytic@translation)))
rot=np.array([[np.cos(.13),-np.sin(.13),0],[np.sin(.13),np.cos(.13),0],[0,0,1.]])
rxyz=xyz@rot.T;rotpath=ROOT/'rotated.npz'
if rotpath.exists():rotated=load_anchored_roks_point(rotpath,anchor,request_hash=request,coordinates_angstrom=rxyz)
else:
 rotated=evaluate_anchored_roks(calc,rxyz,anchor,reference_result=central,gradient=True,**options)
 save_anchored_roks_point(rotpath,rotated,request_hash=request,coordinates_angstrom=rxyz)
renergy=abs(rotated.energy_hartree-central.energy_hartree)
rgradient=float(np.max(abs(rotated.gradient_hartree_per_bohr-central.gradient_hartree_per_bohr@rot.T)))
report=dict(host=HOST,molecule='nitroxyl',anchor_fingerprint=anchor.fingerprint,
 gradient_max_error_hartree_per_bohr=gerr,hessian_max_error_hartree_per_bohr2=herr,
 nuclear_hessian_raw_symmetry_error=numeric.symmetry_error_hartree_per_bohr2,
 analytic_translation_sum_rule=terr,orbital_response_residual=response_error,
 rigid_rotation_energy_error_hartree=renergy,rigid_rotation_gradient_error=rgradient,
 nuclear_displacements_used_in_analytic_method=0,
 independent_check='central differences of retained analytic gradients and energies',
 scope='frozen reference W,U,n_anchor and energy offset; no geometry-dependent DFT recalibration')
report['passed']=gerr<2e-5 and herr<1e-4 and terr<1e-7 and response_error<1e-8 and renergy<1e-8 and rgradient<1e-6
write(ROOT/'validation.json',report);print(json.dumps(report,indent=2),flush=True)
assert report['passed']
