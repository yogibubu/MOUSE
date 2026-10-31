"""Read-only molecular geometry and ground/CSF orthogonality assessment."""
import importlib.util,json,sys
from pathlib import Path
BASE=Path('/home/enzo/MOUSE-roks-direct-20261030')
spec=importlib.util.spec_from_file_location('p',BASE/'source/qualification/probe.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
from matrix_gfn2.deltascf_derivatives import load_derivative_point
np=p.np

def angle(a,b):return float(np.degrees(np.arccos(np.clip(a@b/np.linalg.norm(a)/np.linalg.norm(b),-1,1))))
for name in sys.argv[1:]:
 branch='ground_replacement_minimum' if name=='acetaldehyde' else 'ground_replacement';root=BASE/'results'/name/branch
 if not (root/'validation.json').exists():continue
 with np.load(root/'accepted_geometry.npz') as d:xyz=d['xyz'].copy()
 rows=[]
 for i in range(2):
  e=load_derivative_point(root/f'excited_{i}.npz',request_hash='ground_replacement_v1',coordinates_angstrom=xyz)
  g=load_derivative_point(root/f'ground_{i+1}.npz',request_hash='ground_replacement_validation',coordinates_angstrom=xyz)
  c=e.common_coefficients;s=e.overlap;ground=g.coefficients[0][:,g.occupations[0]>.5]
  amplitude=0.
  for occ in [e.occupations,e.occupations[::-1]]:
   amplitude+=np.prod([np.linalg.det(ground.T@s@c[:,o>.5]) for o in occ])/np.sqrt(2.)
  opens=np.flatnonzero(e.occupations[0]!=e.occupations[1]);leak=np.sum((ground.T@s@c[:,opens])**2,axis=0)
  rows.append(dict(component=i,ground_csf_overlap=float(abs(amplitude)),open_orbital_ground_occupied_fractions=leak.tolist()))
 geometry={}
 if name=='nitroxyl':
  geometry=dict(NO=float(np.linalg.norm(xyz[0]-xyz[1])),NH=float(np.linalg.norm(xyz[2]-xyz[1])),HNO=angle(xyz[0]-xyz[1],xyz[2]-xyz[1]))
 elif name=='acetaldehyde':
  co=xyz[2]-xyz[0];cc=xyz[1]-xyz[0];ch=xyz[3]-xyz[0];normal=np.cross(cc,ch)
  eta=float(np.degrees(np.arcsin(np.clip(abs(co@normal)/np.linalg.norm(co)/np.linalg.norm(normal),0,1))))
  geometry=dict(CO=float(np.linalg.norm(co)),CC=float(np.linalg.norm(cc)),CH_aldehyde=float(np.linalg.norm(ch)),CCO=angle(cc,co),CCH=angle(cc,ch),eta=eta)
 out=dict(molecule=name,geometry=geometry,orthogonality=rows)
 (root/'molecular_assessment.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
