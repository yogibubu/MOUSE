"""Resume an accepted optimized geometry with checkpointed parallel Hessian columns.

The incomplete serial Hessian is interrupted only after the optimized orbitals,
energy and analytic gradient have been saved and checked. Completed columns
are never recalculated. This is a qualification timing, not serial API timing.
"""
from pathlib import Path
import socket,runpy,sys,time,json,subprocess,os,signal,argparse
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
BASE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101');HERE=BASE/'oracle/MOUSE-closing-validation-20261101';REPAIR=BASE/'oracle/MOUSE-pi-pi-transfer-repair-20261101';runpy.run_path(str(BASE/'source/qualification/remote_bootstrap.py'))
import numpy as np
from dataclasses import replace
from scipy.linalg import null_space
from threadpoolctl import threadpool_limits
from matrix_gfn2.nosi_surface import PreparedNOSIModel,load_nosi_result,save_nosi_result
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
p=argparse.ArgumentParser();p.add_argument('cohort');p.add_argument('name');p.add_argument('--serial-pid',type=int);p.add_argument('--workers',type=int,default=8);a=p.parse_args();folder=HERE/'structures'/a.name;model=PreparedNOSIModel(REPAIR/'prepared_models'/a.cohort/a.name/'model.json')
while not (folder/'optimized.npz').exists():
 if (folder/'result.json').exists():sys.exit(0)
 time.sleep(3)
r=load_nosi_result(folder/'optimized.npz',model);assert max(np.max(abs(s.gradient_hartree_per_bohr)) for s in r.states)<2e-5
if a.serial_pid:
 command=subprocess.run(['ps','-p',str(a.serial_pid),'-o','command='],capture_output=True,text=True).stdout
 if 'optimize_surface.py '+a.cohort+' '+a.name in command:
  elapsed=subprocess.run(['ps','-p',str(a.serial_pid),'-o','etime='],capture_output=True,text=True).stdout.strip();os.kill(a.serial_pid,signal.SIGTERM)
  (folder/'parallel_handoff.json').write_text(json.dumps(dict(serial_pid=a.serial_pid,serial_elapsed=elapsed,optimized_checkpoint_preserved=True,reason='Resume final analytic Hessian as checkpointed parallel columns; no completed Hessian or electronic preparation restarted'),indent=2)+'\n')
started=time.monotonic();workers=[]
for worker in range(a.workers):
 log=(folder/f'hessian_worker_{worker}.log').open('a');process=subprocess.Popen([sys.executable,str(HERE/'minimum_hessian_columns.py'),a.cohort,a.name,'--worker',str(worker),'--workers',str(a.workers)],stdout=log,stderr=subprocess.STDOUT,env=os.environ);workers.append((process,log))
for process,log in workers:
 code=process.wait();log.close();assert code==0,(process.pid,code)
with threadpool_limits(2):
 nr=r.coordinates_angstrom.size;state=r.states[0];columns=[]
 for i in range(nr):
  with np.load(folder/'hessian_columns'/f'{i:03d}.npz') as z:columns.append(z[state.state_id].copy())
 raw=np.column_stack(columns);asym=float(np.max(abs(raw-raw.T)));assert asym<2e-6;h=.5*(raw+raw.T);state=replace(state,hessian_hartree_per_bohr2=h);final=replace(r,states=(state,));save_nosi_result(folder/'final_hessian.npz',final)
 xyz=r.coordinates_angstrom*ANGSTROM_TO_BOHR;mass=np.array([{1:1.00782503223,6:12.,7:14.00307400443,8:15.99491461957}[int(z)] for z in model.anchor.atomic_numbers]);relative=xyz-np.average(xyz,axis=0,weights=mass);root=np.sqrt(mass);rigid=[(root[:,None]*axis).ravel() for axis in np.eye(3)]+[(root[:,None]*np.cross(axis,relative)).ravel() for axis in np.eye(3)];q=null_space(np.array(rigid));repeated=np.repeat(mass,3);hw=h/np.sqrt(repeated[:,None]*repeated[None,:]);values,vectors=np.linalg.eigh(q.T@hw@q);frequencies=np.sign(values)*np.sqrt(abs(values))*5140.487143715828;cart=(q@vectors)/np.sqrt(repeated[:,None]);np.savez_compressed(folder/'normal_modes.npz',frequencies_cm=frequencies,cartesian_modes=cart,coordinates_bohr=xyz,masses_amu=mass,hessian=h)
 report=dict(molecule=a.name,model_fingerprint=model.fingerprint,host=socket.getfqdn(),gradient_max=float(abs(state.gradient_hartree_per_bohr).max()),frequencies_cm=frequencies.tolist(),imaginary_modes=int(sum(frequencies < -20)),minimum_frequency_cm=float(frequencies.min()),coordinates_angstrom=r.coordinates_angstrom.tolist(),atomic_numbers=model.anchor.atomic_numbers.tolist(),optimization_steps=len(json.loads((folder/'history.json').read_text())),optimization_message='converged analytic gradient; resumed saved optimized checkpoint',hessian_seconds=time.monotonic()-started,hessian_workers=a.workers,hessian_scope='checkpointed parallel analytic columns; not a serial public-API timing',hessian_asymmetry=asym,maximum_stationary_residual=r.maximum_stationary_residual,minimum_overlap=min(s.overlap for s in r.states),derivatives='complete analytic anchored-model response',frequency_scaling=1.)
 (folder/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
