from remote_bootstrap import HOST
import json
import numpy as np
from matrix_gfn2 import GFN2Calculator,evaluate_gfn2_roks_singlet
from matrix_gfn2.response_reference import stationary_response_reference
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
for atoms,xyz,name in [(['H','H'],np.array([[0.,0.,0.],[0.,0.,.85]]),'h2'),
    (['C','O','H','H'],np.array([[0.,0.,0.],[1.21,0.,0.],[-.58,.94,0.],[-.58,-.94,0.]]),'formaldehyde')]:
    calc=GFN2Calculator(atoms,temperature_kelvin=0.)
    ground,certificate=stationary_response_reference(calc,xyz)
    state=ground.orbital_state
    alpha=state.alpha_occupations.copy();beta=state.beta_occupations.copy()
    hole=np.flatnonzero(alpha>.5)[-1];particle=np.flatnonzero(alpha<.5)[0]
    alpha[hole]=0;alpha[particle]=1
    r=evaluate_gfn2_roks_singlet(calc,xyz,alpha,beta,ground_state=ground,gradient=True)
    print(name,'E',r.energy_hartree,'S2',r.spin_squared,'res',r.orbital_residual,
          'eval',r.evaluations,'gradient',r.gradient_hartree_per_bohr,flush=True)
    v=np.arange(1,xyz.size+1,dtype=float).reshape(xyz.shape);v/=np.linalg.norm(v)
    step=1e-4;energies=[]
    for sign in [-1,1]:
        rr=evaluate_gfn2_roks_singlet(calc,xyz+sign*step*v/ANGSTROM_TO_BOHR,alpha,beta,reference_result=r)
        energies.append(rr.energy_hartree)
    fd=(energies[1]-energies[0])/(2*step);analytic=float(np.sum(v*r.gradient_hartree_per_bohr))
    print(json.dumps(dict(host=HOST,molecule=name,energy=r.energy_hartree,
        gradient_direction=analytic,fd_direction=fd,error=abs(fd-analytic))),flush=True)
