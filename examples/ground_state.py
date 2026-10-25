"""Ground-state energy, gradient and analytic ALPB Hessian of water."""
import numpy as np
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.response_reference import stationary_response_reference
atoms=['O','H','H']
x=np.array([[0.,0.,0.],[.9572,0.,0.],[-.23999,.92663,0.]])
c=GFN2Calculator(atoms,temperature_kelvin=0.,alpb_solvent='water')
g,certificate=stationary_response_reference(c,x)
hessian=c.ground_state_hessian_operator(g.orbital_state).materialize(symmetrize=False)
print('GFN2-ALPB water energy / hartree:',g.energy_hartree)
print('Gradient / hartree per bohr:',g.gradient_hartree_per_bohr)
print('Analytic Hessian shape:',hessian.shape)
