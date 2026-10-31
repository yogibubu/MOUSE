"""Checkpoint-only diagnosis: active pair versus closed-shell spectators."""
import importlib.util,json
from pathlib import Path
from types import SimpleNamespace
BASE=Path('/home/enzo/MOUSE-hno-fock-20261030');OLD=Path('/home/enzo/MOUSE-roks-following-20261030')
spec=importlib.util.spec_from_file_location('p',BASE/'source/qualification/probe.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
from matrix_link.scan import QMScanBackend
from matrix_link.composite_roks_component import restore_accepted
from matrix_gfn2 import load_response_reference
from matrix_gfn2.nonlocal_response import TransportedOperators
np=p.np

def q(result,frame):
 vals,v=np.linalg.eigh(result.overlap);root=(v*np.sqrt(vals))@v.T
 return TransportedOperators(result.reference_state,frame,np.eye(len(vals))[None]).u.T@root@result.common_coefficients

def compare(a,b,frame):
 qa,qb=q(a,frame),q(b,frame);occ=a.occupations
 active=np.flatnonzero(occ[0]!=occ[1]);closed=np.flatnonzero(np.sum(occ,axis=0)==2)
 x=qa[:,active].T@qb[:,active]
 pair=float(abs(x[0,0]*x[1,1]+x[0,1]*x[1,0]))
 sv=np.linalg.svd(qa[:,closed].T@qb[:,closed],compute_uv=False)
 return dict(active_pair=pair,active_subspace_singular_values=np.linalg.svd(x,compute_uv=False).tolist(),
    closed_subspace_singular_values=sv.tolist(),closed_determinant_magnitude=float(np.prod(sv)))

def local_character(r):
 s=r.reference_state;vals,v=np.linalg.eigh(r.overlap);lowdin=(v*np.sqrt(vals))@v.T@r.common_coefficients
 coords=s.coordinates_bohr;z=s.atomic_numbers;active=np.flatnonzero(r.occupations[0]!=r.occupations[1]);rows=[]
 carbonyls=[]
 for oxygen in np.flatnonzero(z==8):
  carbon=min(np.flatnonzero(z==6),key=lambda c:np.linalg.norm(coords[c]-coords[oxygen]))
  heavy=[i for i in range(len(z)) if z[i]>1 and i not in [oxygen,carbon]]
  neighbor=min(heavy,key=lambda i:np.linalg.norm(coords[i]-coords[carbon]))
  normal=np.cross(coords[oxygen]-coords[carbon],coords[neighbor]-coords[carbon]);normal/=np.linalg.norm(normal)
  carbonyls.append((int(oxygen),int(carbon),normal))
 for orbital in active:
  out=0.;total=0.;oxygen_population=0.
  for oxygen,carbon,normal in carbonyls:
   oxygen_population+=float(np.sum(lowdin[np.asarray(s.ao_atom)==oxygen,orbital]**2))
   for atom in [oxygen,carbon]:
    ids=[i for i,(a,shell) in enumerate(zip(s.ao_atom,s.ao_shell)) if a==atom and str(shell).endswith('p')]
    if len(ids)!=3:raise ValueError('expected one carbonyl p shell')
    vector=lowdin[ids,orbital];out+=float((vector@normal)**2);total+=float(vector@vector)
  rows.append(dict(orbital=int(orbital),role='hole' if r.occupations[0,orbital]<r.occupations[1,orbital] else 'particle',oxygen_lowdin_population=oxygen_population,
    carbonyl_p_population=total,carbonyl_p_out_of_local_plane_fraction=out/total if total else None))
 return rows

rows=[]
with p.threadpool_limits(limits=2):
 for name in ['acetaldehyde','maleimide']:
  root=OLD/'results'/name;data=json.loads((root/'prepared.json').read_text())
  backend=QMScanBackend(name='matrix_gfn2',electronic_state=1,native_excited_method='roks_50_50',deltascf_promotion=tuple(data['promotion']),
   native_roks_composite_response_paths=tuple(data['response_paths']),native_roks_composite_saddle_indices=tuple(data['saddle_indices']),
   native_roks_composite_following='character',native_roks_composite_ground_recovery='accepted_retry')
  comp=SimpleNamespace(backend=backend,run_dir=root/'ground_recovery/roks_50_50',reference=None,anchor=None);restore_accepted(comp)
  result=comp.reference[0];models=[load_response_reference(path) for path in backend.native_roks_composite_response_paths];frame=models[0].anchor_coordinates_bohr
  rows.append(dict(molecule=name,component_full_csf_overlap=result.component_csf_overlap,
   cross_component=compare(*result.components,frame),anchor_comparisons=[compare(a,b,frame) for a,b in zip(comp.anchor[0].components,result.components)],
   anchor_local_character=[local_character(a) for a in comp.anchor[0].components],terminal_local_character=[local_character(a) for a in result.components]))
 (BASE/'other_molecule_tracking.json').write_text(json.dumps(dict(host=p.socket.getfqdn(),diagnostic_only=True,thresholds_unchanged=True,rows=rows),indent=2)+'\n')
 for r in rows:print(json.dumps(r),flush=True)
