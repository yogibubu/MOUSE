"""Compact TDA derivative API on formaldehyde; no absolute optical calibration."""
import numpy as np
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.response_reference import stationary_response_reference
from matrix_gfn2.stda import compact_xtb_tda
atoms=['C','O','H','H']
x=np.array([[0.,0.,0.],[1.21,0.,0.],[-.58,.94,0.],[-.58,-.94,0.]])
c=GFN2Calculator(atoms,temperature_kelvin=0.)
g,certificate=stationary_response_reference(c,x)
r=compact_xtb_tda(g.orbital_state,virtual_count=4,roots=3)
op=c.closed_shell_stda_hessian(g.orbital_state,excited=r,root=0)
v=np.arange(x.size,dtype=float);v/=np.linalg.norm(v)
print('Native compact TDA gaps / eV:',r.excitation_energies_hartree*27.211386245988)
print('Analytic excitation gradient / hartree per bohr:',op.excitation_gradient_hartree_per_bohr)
print('Analytic total-state Hessian action / hartree per bohr squared:',op.matvec(v))
