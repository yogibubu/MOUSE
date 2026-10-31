import importlib.util,json
from pathlib import Path
BASE=Path('/home/enzo/MOUSE-hno-fock-20261030')
spec=importlib.util.spec_from_file_location('p',BASE/'source/qualification/probe.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
np=p.np
import matrix_gfn2.calculator as module
from matrix_gfn2.response_reference import stationary_response_reference
from scipy.linalg import eigh
original=module.GFN2OrbitalState
records=[]
def inspect_state(**kw):
 c,s,f,e=[kw[k] for k in ['coefficients','overlap','fock','orbital_energies_hartree']]
 gram=c.T@s@c;err=float(np.max(abs(gram-np.eye(len(s)))))
 records.append(dict(orthogonality_error=err,eigen_residual=float(np.max(abs(f@c-s@c*e))),overlap_eigenvalues=np.linalg.eigvalsh(s).tolist(),energies=e.tolist()))
 if err>1e-7:np.savez_compressed(BASE/'failed_orthogonality.npz',c=c,s=s,f=f,e=e,xyz=kw['coordinates_bohr']/p.ANGSTROM_TO_BOHR)
 return original(**kw)
module.GFN2OrbitalState=inspect_state
with p.threadpool_limits(limits=2):
 # Load the newly accepted point, with its original anchor and fingerprints.
 from types import SimpleNamespace
 from matrix_link.composite_roks_component import restore_accepted
 comp=SimpleNamespace(backend=p.b,run_dir=BASE/'results/nitroxyl/resume/roks_50_50',reference=None,anchor=None);restore_accepted(comp)
 xyz=comp.reference[0].components[0].reference_state.coordinates_bohr/p.ANGSTROM_TO_BOHR
 for tolerance in [1e-6,1e-10]:
  calc=p.GFN2Calculator(p.p['atoms'],temperature_kelvin=0.,maximum_scf_iterations=500,scf_energy_tolerance_hartree=tolerance)
  try:r,d=stationary_response_reference(calc,xyz);print('PASS',tolerance,d)
  except Exception as exc:print('FAIL',tolerance,str(exc))
 (BASE/'orthogonality_probe.json').write_text(json.dumps(records,indent=2)+'\n')
 print(json.dumps(records,indent=2))
