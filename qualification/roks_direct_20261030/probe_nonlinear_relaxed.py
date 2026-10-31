"""Energy-only feasibility away from the calibration point.
XC remainder transported with a positive density congruence in Lowdin gauge;
old quadratic baseline unchanged. No nuclear-derivative claim is made.
"""
import importlib.util,json
from pathlib import Path
from dataclasses import replace
BASE=Path('/home/enzo/MOUSE-roks-direct-20261030')
spec=importlib.util.spec_from_file_location('m',BASE/'source/qualification/probe_nonlinear_xc.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
p=m.p;np=m.np
from matrix_gfn2.nonlocal_response import TransportedOperators
from matrix_gfn2.deltascf import project_delta_orbitals
from scipy.linalg import fractional_matrix_power
with p.threadpool_limits(limits=2):
 name='acetaldehyde';root=BASE/'results'/name/'nonlinear_relaxed';root.mkdir(exist_ok=True)
 if (root/'result.json').exists():raise SystemExit(0)
 source=BASE/'results'/name/'ground_replacement_minimum'
 with np.load(source/'accepted_geometry.npz') as d:xyz=d['xyz'].copy()
 cp=Path('/home/enzo/MOUSE-roks-50-50-optimization-20261030/results')/name/'full_response/seed.npz'
 with np.load(cp) as d:vertical=d['coordinates_angstrom'].copy()
 old=m.load_derivative_point(cp,request_hash='retained_vertical',coordinates_angstrom=vertical,require_gradient=False);s=old.reference_state
 data=json.loads((BASE/'results'/name/'prepared.json').read_text());refs=[p.load_response_reference(x) for x in data['response_paths']]
 folder=Path(json.loads((m.OLD/'prepared_B'/name/'case.json').read_text())['reference'])
 with np.load(m.OLD/'results/full_response'/name/'embedding.npz') as d:lift=d['embedding'].copy()
 with np.load(folder/'reference.npz') as d:c=d['coefficients'];occ=d['occupations']
 core=c[:,np.flatnonzero(occ==2)[:int(np.sum(old.core_electrons)//2)]];mol=m.scf.chkfile.load_mol(str(folder/'backend.chk'))
 residual=m.XCResidual(mol,lift,core,s,refs[0],BASE/'results'/name/'nonlinear_xc/grid_features.npz')
 calc=p.GFN2Calculator(data['atoms'],temperature_kelvin=0.)
 ctx0=p.prepare_gfn2_deltascf(calc,xyz,orbital_reference=s,spin_reference=refs[0])
 def root_metric(s,power):
  val,vec=np.linalg.eigh(s);return (vec*val**power)@vec.T
 inverse=root_metric(s.overlap,-.5);current=root_metric(ctx0.state.overlap,.5)
 operators=inverse@residual.b@inverse
 transported=TransportedOperators(ctx0.state,s.coordinates_bohr,operators)
 residual.b=current@transported.a@current;residual.cache={}
 rows=[]
 for i,ref in enumerate(refs):
  ctx=p.prepare_gfn2_deltascf(calc,xyz,orbital_reference=s,spin_reference=ref)
  ground,ground_cache=m.solve(ctx,residual,ctx.state.coefficients,np.array([s.alpha_occupations,s.beta_occupations]),0,root/f'ground_{i}_latest.npz')
  seed=project_delta_orbitals(old,ctx)[0]
  for index in [0,1]:
   try:
    result,cache=m.solve(ctx,residual,seed,old.occupations,index,root/f'excited_{i}_index{index}_latest.npz')
    descriptor=replace(old,common_coefficients=result.point,overlap=ctx.state.overlap,reference_state=ctx.state)
    pair=m.active_orbital_character(old,descriptor,s.coordinates_bohr,allow_different_response=True)['transition']
    a=ground.point[:,s.alpha_occupations>.5];amplitude=0.
    for o in [old.occupations,old.occupations[::-1]]:
     amplitude+=np.prod([np.linalg.det(a.T@ctx.state.overlap@result.point[:,oo>.5]) for oo in o])/np.sqrt(2.)
    row=dict(component=i,index=index,gap_ev=(result.energy-ground.energy)*27.211386245988,ground_overlap=float(abs(amplitude)),initial_pair_overlap=pair,orbital_derivative_checks=cache['derivative_checks'])
   except Exception as exc:row=dict(component=i,index=index,error=str(exc))
   rows.append(row);print(json.dumps(row),flush=True);(root/'partial.json').write_text(json.dumps(rows,indent=2)+'\n')
 (root/'result.json').write_text(json.dumps(dict(rows=rows,nuclear_derivatives=False,density_transport='positive Lowdin congruence for the XC remainder only',baseline='unchanged quadratic response'),indent=2)+'\n')
