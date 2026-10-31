import importlib.util,json
from pathlib import Path
from dataclasses import replace
BASE=Path('/home/enzo/MOUSE-roks-direct-20261030')
spec=importlib.util.spec_from_file_location('p',BASE/'source/qualification/probe.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
from matrix_gfn2.deltascf_derivatives import load_derivative_point,save_derivative_point
from scipy.linalg import expm
np=p.np
with p.threadpool_limits(limits=2):
 name='nitroxyl';data=json.loads((BASE/'results'/name/'prepared.json').read_text());rows=[]
 for i,profile in enumerate(['full_response','qualified_spectral_response']):
  cp=Path('/home/enzo/MOUSE-roks-50-50-optimization-20261030/results')/name/profile/'seed.npz'
  with np.load(cp) as d:xyz=d['coordinates_angstrom'].copy()
  old=load_derivative_point(cp,request_hash='retained_vertical',coordinates_angstrom=xyz,require_gradient=False)
  def characterize(r):
   s=r.reference_state;v,u=np.linalg.eigh(s.overlap);root=(u*np.sqrt(v))@u.T
   ids=[]
   for a in range(len(s.atomic_numbers)):
    for label in dict.fromkeys(s.ao_shell):
     group=[j for j,(aa,l) in enumerate(zip(s.ao_atom,s.ao_shell)) if aa==a and l==label]
     if len(group)==3:ids.append(group[1])
   opens=np.flatnonzero(r.occupations[0]!=r.occupations[1]);q=root@r.common_coefficients[:,opens]
   return dict(open_indices=opens.tolist(),out_of_plane_fraction=np.sum(q[ids]**2,axis=0).tolist())
  rows.append(dict(component=i,branch='original',character=characterize(old)))
  if i==0:
   opens=np.flatnonzero(old.occupations[0]!=old.occupations[1]);k=np.zeros_like(old.common_coefficients);a,b=opens;k[a,b]=np.pi/4;k[b,a]=-np.pi/4
   seed=replace(old,common_coefficients=old.common_coefficients@expm(k))
   calc=p.GFN2Calculator(data['atoms'],temperature_kelvin=0.);ref=p.load_response_reference(data['response_paths'][i])
   new=p.evaluate_gfn2_roks_singlet(calc,xyz,*old.occupations,spin_reference=ref,reference_result=seed,reference_policy='geometry_only',orbital_solver='gmf',saddle_index=1,maximum_iterations=200,orbital_tolerance=1e-10,gradient=True)
   save_derivative_point(BASE/'results'/name/'fr_index1.npz',new,request_hash='fr_index1',coordinates_angstrom=xyz)
   rows.append(dict(component=i,branch='open_rotated_index1',character=characterize(new),energy_change_ev=(new.energy_hartree-old.energy_hartree)*27.211386245988))
 print(json.dumps(rows,indent=2));(BASE/'results'/name/'open_pair_diagnosis.json').write_text(json.dumps(rows,indent=2)+'\n')
