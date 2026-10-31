"""Independent native spGFN2 energy/gradient checks, run only on verified Oracle."""
from remote_bootstrap import BASE,HOST
import json,re,subprocess
import numpy as np
from matrix_gfn2 import GFN2Calculator,evaluate_gfn2_deltascf
from matrix_gfn2.response_reference import stationary_response_reference
records=[]
for name,atoms,xyz in [('h2',['H','H'],np.array([[0.,0.,0.],[0.,0.,.85]])),
 ('formaldehyde',['C','O','H','H'],np.array([[0.,0.,0.],[1.21,0.,0.],[-.58,.94,0.],[-.58,-.94,0.]]))]:
    folder=BASE/'results'/'orca'/name;folder.mkdir(parents=True,exist_ok=True)
    inp=folder/'triplet.inp';out=folder/'triplet.out'
    text='! Native-spGFN2-xTB TightSCF EnGrad NoAutoStart\n%maxcore 512\n%scf\n SmearTemp 0\n UseXTBMixer false\n MaxIter 300\nend\n* xyz 0 3\n'
    text+='\n'.join(f'{a} {x:.12f} {y:.12f} {z:.12f}' for a,(x,y,z) in zip(atoms,xyz))+'\n*\n'
    if not inp.exists():inp.write_text(text)
    if not out.exists() or 'ORCA TERMINATED NORMALLY' not in out.read_text():
        with out.open('w') as stream:
            subprocess.run(['/home/enzo/orca_6_1_1/orca',inp.name],cwd=folder,stdout=stream,stderr=subprocess.STDOUT,check=True)
    output=out.read_text();energy=float(re.findall(r'FINAL SINGLE POINT ENERGY\s+([-\d.]+)',output)[-1])
    lines=(folder/'triplet.engrad').read_text().splitlines()
    numbers=[float(token) for x in lines if x.strip() and not x.lstrip().startswith('#') for token in x.split()]
    n=int(numbers[0]);gradient=np.array(numbers[2:2+3*n]).reshape(n,3)
    calc=GFN2Calculator(atoms,temperature_kelvin=0.)
    ground,_=stationary_response_reference(calc,xyz)
    state=ground.orbital_state;alpha=state.alpha_occupations.copy();beta=state.beta_occupations.copy()
    hole=np.flatnonzero(beta>.5)[-1];particle=np.flatnonzero(alpha<.5)[0]
    beta[hole]=0;alpha[particle]=1
    result=evaluate_gfn2_deltascf(calc,xyz,alpha,beta,ground_state=ground,gradient=True,
        density_tolerance=1e-10,orbital_tolerance=1e-10,energy_tolerance=1e-12,maximum_iterations=1000)
    record=dict(host=HOST,molecule=name,orca_energy_hartree=energy,mouse_energy_hartree=result.energy_hartree,
        energy_difference_hartree=result.energy_hartree-energy,
        gradient_max_difference_hartree_per_bohr=float(np.max(np.abs(result.gradient_hartree_per_bohr-gradient))),
        mouse_spin_squared=result.spin_squared,orca_version='6.1.1',method='native-spGFN2-xTB triplet')
    records.append(record);print(json.dumps(record),flush=True)
(BASE/'results'/'orca_spin_comparison.json').write_text(json.dumps(records,indent=2)+'\n')
