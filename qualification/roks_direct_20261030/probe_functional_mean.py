"""Distinct research variant: average functionals BEFORE orbital optimization.
Fixed 50/50, same lambda; no fitted offsets. Not the production energy mean.
"""
import importlib.util,json,sys
from pathlib import Path
from dataclasses import replace
from scipy.linalg import block_diag
BASE=Path('/home/enzo/MOUSE-roks-direct-20261030')
spec=importlib.util.spec_from_file_location('p',BASE/'source/qualification/probe.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
from matrix_gfn2.deltascf_derivatives import load_derivative_point,save_derivative_point
from matrix_gfn2.restricted_ground import evaluate_restricted_ground
from matrix_gfn2 import save_response_reference
from matrix_gfn2.roks_tracking import active_orbital_character
np=p.np
with p.threadpool_limits(limits=2):
 name=sys.argv[1];root=BASE/'results'/name;data=json.loads((root/'prepared.json').read_text());refs=[p.load_response_reference(x) for x in data['response_paths']]
 a,b=refs
 for field in ['atomic_numbers','orbital_shell','anchor_coordinates_bohr']:
  assert np.array_equal(getattr(a,field),getattr(b,field))
 blend=replace(a,source_model='fixed_functional_mean.v1:'+a.fingerprint+':'+b.fingerprint,
  operators=np.concatenate([a.operators,b.operators]),
  pair_charge_hartree=.5*block_diag(a.pair_charge_hartree,b.pair_charge_hartree),
  pair_spin_hartree=.5*block_diag(a.pair_spin_hartree,b.pair_spin_hartree),
  pair_reference_moments=np.concatenate([a.pair_reference_moments,b.pair_reference_moments]),
  pair_linear_hartree=.5*np.concatenate([a.pair_linear_hartree,b.pair_linear_hartree]))
 outroot=root/'functional_mean';outroot.mkdir(exist_ok=True);save_response_reference(outroot/'response.npz',blend)
 rows=[]
 for geom in ['vertical','relaxed']:
  states=[]
  if geom=='relaxed':
   folder=root/'ground_replacement_minimum'
   if not folder.exists():folder=root/'ground_replacement'
   with np.load(folder/'accepted_geometry.npz') as d:xyz=d['xyz'].copy()
   states=[load_derivative_point(folder/f'excited_{i}.npz',request_hash='ground_replacement_v1',coordinates_angstrom=xyz) for i in range(2)]
  else:
   for profile in ['full_response','qualified_spectral_response']:
    cp=Path('/home/enzo/MOUSE-roks-50-50-optimization-20261030/results')/name/profile/'seed.npz'
    with np.load(cp) as d:xyz=d['coordinates_angstrom'].copy()
    states.append(load_derivative_point(cp,request_hash='retained_vertical',coordinates_angstrom=xyz,require_gradient=False))
  calc=p.GFN2Calculator(data['atoms'],temperature_kelvin=0.)
  ground=evaluate_restricted_ground(calc,xyz,template=states[0].reference_state,spin_reference=blend)
  for seedindex,old in enumerate(states):
   for index in [0,1,2]:
    try:
     seed=replace(old,spin_reference_fingerprint=blend.fingerprint)
     result=p.evaluate_gfn2_roks_singlet(calc,xyz,*old.occupations,spin_reference=blend,reference_result=seed,minimum_state_overlap=0.,reference_policy='geometry_only',orbital_solver='gmf',saddle_index=index,maximum_iterations=150,orbital_tolerance=1e-10,gradient=True)
     c=result.common_coefficients;g=ground.coefficients[0][:,ground.occupations[0]>.5];amplitude=0.
     for occ in [result.occupations,result.occupations[::-1]]:amplitude+=np.prod([np.linalg.det(g.T@result.overlap@c[:,o>.5]) for o in occ])/np.sqrt(2.)
     row=dict(geometry=geom,seed_component=seedindex,index=index,gap_ev=(result.energy_hartree-ground.energy_hartree)*27.211386245988,ground_csf_overlap=float(abs(amplitude)),initial_pair_overlap=active_orbital_character(old,result,blend.anchor_coordinates_bohr,allow_different_response=True)['transition'])
     save_derivative_point(outroot/f'{geom}_{seedindex}_index{index}.npz',result,request_hash='functional_mean_probe',coordinates_angstrom=xyz)
    except Exception as exc:row=dict(geometry=geom,seed_component=seedindex,index=index,error=str(exc))
    rows.append(row);print(json.dumps(row),flush=True);(outroot/'probe.json').write_text(json.dumps(rows,indent=2)+'\n')
