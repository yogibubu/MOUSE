"""Remote fixed-density native/response decomposition at the retained HNO path."""
import os,sys,socket,json
from pathlib import Path
from types import SimpleNamespace
BASE=Path('/home/enzo/MOUSE-hno-fock-20261030');OLD=Path('/home/enzo/MOUSE-roks-following-20261030')
assert socket.getfqdn()=='oracle.sns.it'
os.environ['MOUSE_VALIDATION_ROOT']=str(BASE)
sys.path.insert(0,'/home/enzo/MOUSE-roks-independent-20261030');import remote_bootstrap
for p in (BASE/'source/packages').glob('*/src'):sys.path.insert(0,str(p))
import numpy as np
from threadpoolctl import threadpool_limits
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from matrix_gfn2.response_reference import stationary_response_reference
from matrix_link.scan import QMScanBackend
from matrix_link.composite_roks_component import restore_accepted
from matrix_gfn2.multipole import anisotropic_energy_and_moment_potentials,dense_scc_fock
p=json.loads((OLD/'results/nitroxyl/prepared.json').read_text())
b=QMScanBackend(name='matrix_gfn2',electronic_state=1,native_excited_method='roks_50_50',deltascf_promotion=tuple(p['promotion']),native_roks_composite_response_paths=tuple(p['response_paths']),native_roks_composite_saddle_indices=tuple(p['saddle_indices']),native_roks_composite_following='character_adaptive_index')
c=SimpleNamespace(backend=b,run_dir=OLD/'results/nitroxyl/resume/roks_50_50',reference=None,anchor=None);restore_accepted(c)
r=c.reference[0];xyz=r.components[0].reference_state.coordinates_bohr/ANGSTROM_TO_BOHR
v=-r.gradient_hartree_per_bohr;v/=np.linalg.norm(v)

def maximum(a,b):return float(np.max(abs(np.asarray(a)-np.asarray(b))))

def probe(step):
 calc=GFN2Calculator(p['atoms'],temperature_kelvin=0.)
 x=xyz+step/ANGSTROM_TO_BOHR*v
 try:ground,diag=stationary_response_reference(calc,x)
 except RuntimeError as exc:
  ground=calc.evaluate(x,retain_orbitals=True,compute_cm5=False,compute_bond_orders=False);diag=dict(error=str(exc))
 s=ground.orbital_state;k=calc.density_response_kernel(s);model=calc._model(s.coordinates_bohr,relocate_basis=True);core=model[4]
 density=(s.coefficients*(s.alpha_occupations+s.beta_occupations))@s.coefficients.T
 native=calc._active_native_context
 fn,en=native.scf_state_fock(density)
 obs,an,d4,energy=native.last_scf_state()
 obs=tuple(np.asarray(t).copy() for t in obs);an=tuple(np.asarray(t).copy() for t in an);d4=tuple(np.asarray(t).copy() for t in d4)
 py_an=anisotropic_energy_and_moment_potentials(k.reference_multipoles,s.coordinates_bohr,k.anisotropic_parameters,geometry=k.anisotropic_geometry)
 py_d4=k._d4(k.reference_multipoles.charges)
 fp=core+k.density_dependent_fock(np.zeros_like(s.fock))
 rec=dense_scc_fock(core,s.overlap,k.operators,k.partition.orbital_shell,k._shell_potential(obs[0]),*an[1:],d4[1])
 row=dict(step_bohr=step,diagnostics=diag,native_response=maximum(fn,fp),native_export=maximum(fn,s.fock),reconstructed_native=maximum(fn,rec),
  observable_errors=[maximum(a,b) for a,b in zip(obs,[k.reference_shell_charges,k.reference_multipoles.charges,k.reference_multipoles.dipoles_bohr,k.reference_multipoles.quadrupoles_bohr2])],
  anisotropic_errors=[maximum(a,b) for a,b in zip(an,py_an)],d4_energy_difference=float(d4[0]-py_d4.energy_hartree),d4_charge_difference=maximum(d4[1],py_d4.charge_derivative_hartree))
 np.savez_compressed(BASE/f'decomposition_{step}.npz',coordinates_angstrom=x,density=density,native_fock=fn,response_fock=fp,reconstructed_native=rec,native_d4_charge=d4[1],python_d4_charge=py_d4.charge_derivative_hartree)
 return row
if __name__=='__main__':
 with threadpool_limits(limits=2):
  rows=[]
  for step in [0.,1e-4,2e-4,1e-3]:
   rows.append(probe(step));print(json.dumps(rows[-1]),flush=True)
   (BASE/'decomposition.json').write_text(json.dumps(dict(host=socket.getfqdn(),rows=rows),indent=2)+'\n')
