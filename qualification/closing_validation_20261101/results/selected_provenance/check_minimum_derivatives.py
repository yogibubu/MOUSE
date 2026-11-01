"""Independent scalar-energy validation at the final optimized minima."""
from pathlib import Path
import socket,runpy,sys,json,argparse,time
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
BASE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101');HERE=BASE/'oracle/MOUSE-closing-validation-20261101';REPAIR=BASE/'oracle/MOUSE-pi-pi-transfer-repair-20261101';runpy.run_path(str(BASE/'source/qualification/remote_bootstrap.py'))
import numpy as np
from threadpoolctl import threadpool_limits
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.nosi_surface import PreparedNOSIModel,load_nosi_result
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
p=argparse.ArgumentParser();p.add_argument('cohort');p.add_argument('name');a=p.parse_args();folder=HERE/'structures'/a.name
while not (folder/'result.json').exists():time.sleep(3)
with threadpool_limits(2):
 model=PreparedNOSIModel(REPAIR/'prepared_models'/a.cohort/a.name/'model.json');r=load_nosi_result(folder/'final_hessian.npz',model);state=r.states[0];nr=r.coordinates_angstrom.size;calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in model.anchor.atomic_numbers],temperature_kelvin=0.)
 with np.load(folder/'normal_modes.npz') as z:soft=z['cartesian_modes'][:,0].copy()
 soft/=np.linalg.norm(soft);mixed=np.sin(np.arange(nr)+.37).reshape(-1,3);mixed-=mixed.mean(0);mixed=mixed.ravel()/np.linalg.norm(mixed);out=folder/'finite_energy';out.mkdir(exist_ok=True);result=[]
 for index,vector in enumerate([soft,mixed]):
  estimates=[]
  for step in [1e-3,5e-4]:
   energies=[]
   for sign in [-1,1]:
    path=out/f'd{index}_h{step:.8f}_s{sign}.json'
    if path.exists():value=json.loads(path.read_text())
    else:
     v=model.evaluate(calc,r.coordinates_angstrom+sign*step*vector.reshape(-1,3)/ANGSTROM_TO_BOHR,properties=('energy',),reference_result=r);s=v.states[0];value=dict(energy=s.energy_hartree,overlap=s.overlap,model=model.fingerprint,maximum_stationary_residual=v.maximum_stationary_residual);path.write_text(json.dumps(value,indent=2)+'\n')
    assert value['model']==model.fingerprint;energies.append(value['energy'])
   estimates.append(dict(gradient=(energies[1]-energies[0])/(2*step),curvature=(energies[1]+energies[0]-2*state.energy_hartree)/step**2))
  gradient=(4*estimates[1]['gradient']-estimates[0]['gradient'])/3;curvature=(4*estimates[1]['curvature']-estimates[0]['curvature'])/3;ga=float(state.gradient_hartree_per_bohr.ravel()@vector);ha=float(vector@state.hessian_hartree_per_bohr2@vector);result.append(dict(direction='softest harmonic mode' if index==0 else 'mixed internal',analytic_gradient=ga,analytic_curvature=ha,richardson_gradient=gradient,richardson_curvature=curvature,gradient_error=abs(gradient-ga),hessian_curvature_error=abs(curvature-ha),steps_bohr=[1e-3,5e-4],raw=estimates))
 report=dict(molecule=a.name,model=model.fingerprint,rows=result,maximum_gradient_error=max(r['gradient_error'] for r in result),maximum_curvature_error=max(r['hessian_curvature_error'] for r in result),finite_differences='validation only; production gradient and Hessian analytic');(folder/'derivative_certificate.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));assert report['maximum_gradient_error']<1e-7 and report['maximum_curvature_error']<2e-6
