"""Independent two-step molecular validation, preserving all five components."""
import importlib.util,json,sys,time
from pathlib import Path
BASE=Path('/home/enzo/MOUSE-roks-direct-20261030')
spec=importlib.util.spec_from_file_location('p',BASE/'source/qualification/probe.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
from matrix_gfn2 import combine_roks_results
from matrix_gfn2.deltascf_derivatives import load_derivative_point,save_derivative_point
from matrix_gfn2.ground_referenced_roks import reference_roks_to_native_ground,ground_referenced_roks_analytic_hessian
np=p.np

def main(name,branch):
 root=BASE/'results'/name/branch
 if (root/'validation.json').exists():return
 data=json.loads((BASE/'results'/name/'prepared.json').read_text());complete=json.loads((root/'complete.json').read_text());assert complete['status']=='stationary'
 refs=[p.load_response_reference(x) for x in data['response_paths']];calc=p.GFN2Calculator(data['atoms'],temperature_kelvin=0.)
 calc.evaluate=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('unexpected SCC'))
 with np.load(root/'accepted_geometry.npz') as d:xyz=d['xyz'].copy()
 ex=[load_derivative_point(root/f'excited_{i}.npz',request_hash='ground_replacement_v1',coordinates_angstrom=xyz) for i in range(2)]
 composite=combine_roks_results(*ex,component_tracking='active_pair')
 central=reference_roks_to_native_ground(calc,xyz,refs,excited=composite)
 assert abs(central.energy_hartree-complete['energy'])<1e-10
 for i,r in enumerate(central.grounds):save_derivative_point(root/f'ground_{i}.npz',r.reference,request_hash='ground_replacement_validation',coordinates_angstrom=xyz)
 (root/'ground_certificates.json').write_text(json.dumps([dict(minimum_curvature=r.minimum_orbital_curvature_hartree,index=r.orbital_saddle_index) for r in central.grounds],indent=2)+'\n')
 from types import SimpleNamespace
 with np.load(root/'analytic_hessian.npz') as d:h=SimpleNamespace(hessian_hartree_per_bohr2=d['hessian'].copy())
 direction=np.sin(np.arange(xyz.size)+.7).reshape(xyz.shape);direction-=direction.mean(axis=0);direction/=np.linalg.norm(direction)
 checks=[]
 for step in [2e-4,1e-4]:
  values=[]
  for sign in [-1,1]:
   displaced=xyz+sign*step/p.ANGSTROM_TO_BOHR*direction
   states=[p.evaluate_gfn2_roks_singlet(calc,displaced,*r.occupations,spin_reference=ref,reference_result=r,reference_policy='geometry_only',orbital_solver='gmf',saddle_index=r.orbital_saddle_index,orbital_tolerance=1e-10,maximum_iterations=200,gradient=True) for r,ref in zip(ex,refs)]
   result=reference_roks_to_native_ground(calc,displaced,refs,excited=combine_roks_results(*states,component_tracking='active_pair'),reference_result=central)
   values.append(result)
  a,b=values;fd=(b.energy_hartree-a.energy_hartree)/(2*step);dg=(b.gradient_hartree_per_bohr-a.gradient_hartree_per_bohr).ravel()/(2*step)
  checks.append(dict(step_bohr=step,gradient_error=float(abs(fd-np.sum(central.gradient_hartree_per_bohr*direction))),hvp_error=float(np.max(abs(dg-h.hessian_hartree_per_bohr2@direction.ravel())))))
 out=dict(molecule=name,branch=branch,maximum_gradient=float(np.max(abs(central.gradient_hartree_per_bohr))),component_gaps_ev=[g*27.211386245988 for g in central.component_gaps_hartree],spectrum_qualified=central.spectrum_qualified,checks=checks,frequencies_cm=complete['frequencies_cm'],zero_scc=True,zero_fitted_offset=True,hessian_source='retained_completed_analytic_hessian')
 out['numerically_qualified']=out['maximum_gradient']<3e-5 and min(out['frequencies_cm'])>=-20 and all(r['hvp_error']<1e-6 and r['gradient_error']<1e-7 for r in checks)
 out['qualified']=out['numerically_qualified'] and out['spectrum_qualified']
 (root/'validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)

if __name__=='__main__':
 with p.threadpool_limits(limits=2):main(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else 'ground_replacement')
