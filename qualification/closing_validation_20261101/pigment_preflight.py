"""Verified-host native ground-state checkpoint and dense-response size audit."""
from pathlib import Path
import socket,subprocess,runpy,argparse,time,json,resource
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
assert subprocess.check_output(['scutil','--get','ComputerName'],text=True).strip()=='Mac Studio di Vincenzo'
BASE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101');HERE=BASE/'oracle/MOUSE-closing-validation-20261101';runpy.run_path(str(BASE/'source/qualification/remote_bootstrap.py'))
import numpy as np
from threadpoolctl import threadpool_limits
from matrix_gfn2 import GFN2Calculator
from dataclasses import fields
p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args();out=HERE/'pigment_preflight'/a.name;out.mkdir(parents=True,exist_ok=True)
if (out/'result.json').exists():raise SystemExit(0)
lines=(HERE/'pigment_inputs'/(a.name+'.xyz')).read_text().splitlines();atoms=[line.split() for line in lines[2:2+int(lines[0])]];xyz=np.array([[float(v) for v in row[1:4]] for row in atoms]);calc=GFN2Calculator([row[0].capitalize() for row in atoms],temperature_kelvin=0.)
with threadpool_limits(4):
 t=time.monotonic();r=calc.evaluate(xyz,retain_orbitals=True,compute_bond_orders=False);seconds=time.monotonic()-t;s=r.orbital_state;np.savez_compressed(out/'native_ground.npz',**{f.name:np.asarray(getattr(s,f.name)) for f in fields(s) if f.name!='basis'},**{'basis_'+f.name:np.asarray(getattr(s.basis,f.name)) for f in fields(s.basis)});n=len(s.overlap);no=s.electron_count//2;nf=n*(n+1)//2
 result=dict(molecule=a.name,host=socket.getfqdn(),atoms=len(atoms),native_orbitals=n,electrons=s.electron_count,occupied=no,ground_energy_hartree=r.energy_hartree,wall_seconds=seconds,peak_rss_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,dense_ERI_GiB=n**4*8/2**30,complete_pair_chart_operators_GiB=nf*n*n*8/2**30,two_dense_pair_kernels_GiB=2*nf**2*8/2**30,dense_ground_orbital_Hessian_GiB=(no*(n-no))**2*8/2**30,charge=0,spin=0,geometry='Unoptimized source geometry; preflight only',status='Native ground checkpoint complete; no OO excitation energy claimed')
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
