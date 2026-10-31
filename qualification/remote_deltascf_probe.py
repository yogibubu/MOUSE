"""Remote-only molecular derivative probe; prints diagnostics without local fallback."""
import socket, sys, importlib.util, json
from pathlib import Path
host=socket.getfqdn()
if host != 'oracle.sns.it':
    raise RuntimeError(f'Oracle identity check failed: {host}')
base=Path('/home/enzo/MOUSE-deltascf-validation-20261029')
for p in sorted((base/'source/packages').glob('*/src')):
    sys.path.insert(0,str(p))
native_path=next(Path('/home/enzo/nano-MATRIX-current/packages/matrix-gfn2/src/matrix_gfn2').glob('_gfn2_native*.so'))
spec=importlib.util.spec_from_file_location('matrix_gfn2._gfn2_native',native_path)
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native);sys.modules[spec.name]=native
import numpy as np
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.response_reference import stationary_response_reference
from matrix_gfn2.deltascf import evaluate_gfn2_deltascf
atoms=['C','O','H','H']
xyz=np.array([[0.,0.,0.],[1.21,0.,0.],[-.58,.94,0.],[-.58,-.94,0.]])
calc=GFN2Calculator(atoms,temperature_kelvin=0.)
ground,certificate=stationary_response_reference(calc,xyz)
state=ground.orbital_state
alpha=state.alpha_occupations.copy();beta=state.beta_occupations.copy()
hole=np.flatnonzero(alpha>.5)[-1];particle=np.flatnonzero(alpha<.5)[0]
alpha[hole]=0;alpha[particle]=1
print('ground',ground.energy_hartree,certificate,flush=True)
r=evaluate_gfn2_deltascf(calc,xyz,alpha,beta,ground_state=ground,gradient=True,
    maximum_iterations=1000,density_tolerance=1e-10,orbital_tolerance=1e-10,energy_tolerance=1e-12)
print('excited',r.energy_hartree,r.spin_squared,r.iterations,r.gradient_hartree_per_bohr,flush=True)
v=np.arange(1,xyz.size+1,dtype=float).reshape(xyz.shape);v/=np.linalg.norm(v)
step=1e-4
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
energies=[]
for sign in [-1,1]:
    rr=evaluate_gfn2_deltascf(calc,xyz+sign*step*v/ANGSTROM_TO_BOHR,alpha,beta,
        reference_result=r,maximum_iterations=1000,density_tolerance=1e-10,orbital_tolerance=1e-10,energy_tolerance=1e-12)
    energies.append(rr.energy_hartree)
fd=(energies[1]-energies[0])/(2*step)
print(json.dumps(dict(host=host,energy=r.energy_hartree,gap=r.excitation_energy_hartree,
    spin_squared=r.spin_squared,gradient_direction=float(np.sum(v*r.gradient_hartree_per_bohr)),
    fd_direction=fd,error=abs(fd-float(np.sum(v*r.gradient_hartree_per_bohr))))),flush=True)
