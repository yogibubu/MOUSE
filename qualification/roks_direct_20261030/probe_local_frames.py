import importlib.util,json,sys
from pathlib import Path
from dataclasses import replace
BASE=Path('/home/enzo/MOUSE-roks-direct-20261030');sys.path.insert(0,str(BASE/'source/qualification'))
spec=importlib.util.spec_from_file_location('p',BASE/'source/qualification/probe.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
from local_frame_response import LocalReference,LocalTransport,local_pair_score
from matrix_gfn2.restricted_ground import evaluate_restricted_ground
from matrix_gfn2.roks_orbital_response import run_roks_gmf
from matrix_gfn2.deltascf_derivatives import load_derivative_point,save_derivative_point
np=p.np

def root_metric(s,power):
 w,v=np.linalg.eigh(s);return (v*w**power)@v.T
with p.threadpool_limits(limits=2):
 name=sys.argv[1];root=BASE/'results'/name/'local_frames';root.mkdir(exist_ok=True)
 if (root/'result.json').exists():raise SystemExit(0)
 data=json.loads((BASE/'results'/name/'prepared.json').read_text());oldrefs=[p.load_response_reference(x) for x in data['response_paths']];refs=[LocalReference.from_reference(r) for r in oldrefs]
 rows=[]
 for geometry in ['vertical','relaxed']:
  if geometry=='relaxed':
   origin=BASE/'results'/name/'ground_replacement_minimum'
   if not origin.exists():origin=BASE/'results'/name/'ground_replacement'
   with np.load(origin/'accepted_geometry.npz') as d:xyz=d['xyz'].copy()
  for i,ref in enumerate(refs):
   cp=Path('/home/enzo/MOUSE-roks-50-50-optimization-20261030/results')/name/['full_response','qualified_spectral_response'][i]/'seed.npz'
   with np.load(cp) as d:vertical=d['coordinates_angstrom'].copy()
   old=load_derivative_point(cp,request_hash='retained_vertical',coordinates_angstrom=vertical,require_gradient=False)
   if geometry=='vertical':xyz=vertical
   calc=p.GFN2Calculator(data['atoms'],temperature_kelvin=0.);ctx=p.prepare_gfn2_deltascf(calc,xyz,orbital_reference=old.reference_state,spin_reference=ref)
   if geometry=='vertical':
    for triplet in [False,True]:
     f,e=ctx.fock_and_energy(old.densities(triplet=triplet));expected=old.fock_triplet if triplet else old.fock_mixed
     assert np.max(abs(f-expected))<1e-10
   ground=evaluate_restricted_ground(calc,xyz,template=old.reference_state,spin_reference=ref,gradient=True)
   ctx=p.prepare_gfn2_deltascf(calc,xyz,orbital_reference=old.reference_state,spin_reference=ref)
   u=LocalTransport(ctx.state,ref.anchor_coordinates_bohr,ref.operators,ref.clusters).u
   initial=root_metric(ctx.state.overlap,-.5)@u@root_metric(old.overlap,.5)@old.common_coefficients
   try:
    result=run_roks_gmf(ctx,initial,old.occupations,saddle_index=old.orbital_saddle_index,maximum_iterations=150,orbital_tolerance=1e-10)
    result=replace(result,reference_state=ctx.state,core_electrons=old.core_electrons,spin_reference_fingerprint=ref.fingerprint)
    score=local_pair_score(old,result,ref)
    save_derivative_point(root/f'{geometry}_{i}.npz',result,request_hash='local_frame_probe_v1',coordinates_angstrom=xyz)
    row=dict(geometry=geometry,component=i,gap_ev=(result.energy_hartree-ground.energy_hartree)*27.211386245988,local_pair_score=score,orbital_index=result.orbital_saddle_index,clusters=ref.clusters,fingerprint=ref.fingerprint)
   except Exception as exc:row=dict(geometry=geometry,component=i,error=str(exc))
   rows.append(row);print(json.dumps(row),flush=True);(root/'partial.json').write_text(json.dumps(rows,indent=2)+'\n')
 (root/'result.json').write_text(json.dumps(rows,indent=2)+'\n')
