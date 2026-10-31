import importlib.util,json,sys,time
from pathlib import Path
BASE=Path('/home/enzo/MOUSE-roks-direct-20261030')
spec=importlib.util.spec_from_file_location('p',BASE/'source/qualification/probe.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
from matrix_gfn2.roks_orbital_response import run_roks_gmf
from matrix_gfn2.deltascf_derivatives import load_derivative_point
from matrix_gfn2.roks_tracking import active_orbital_character
from dataclasses import replace
np=p.np
with p.threadpool_limits(limits=2):
 for name in sys.argv[1:]:
  data=json.loads((BASE/'results'/name/'prepared.json').read_text());rows=[]
  for i,profile in enumerate(['full_response','qualified_spectral_response']):
   cp=Path('/home/enzo/MOUSE-roks-50-50-optimization-20261030/results')/name/profile/'seed.npz'
   with np.load(cp) as d:xyz=d['coordinates_angstrom'].copy()
   old=load_derivative_point(cp,request_hash='retained_vertical',coordinates_angstrom=xyz,require_gradient=False)
   calc=p.GFN2Calculator(data['atoms'],temperature_kelvin=0.);ref=p.load_response_reference(data['response_paths'][i])
   ctx=p.prepare_gfn2_deltascf(calc,xyz,orbital_reference=old.reference_state,spin_reference=ref)
   start=time.monotonic()
   try:
    # Use original ground orbitals as the promoted starting configuration,
    # rather than asking FR-DO to rediscover an already converged ROKS root.
    result=run_roks_gmf(ctx,old.reference_state.coefficients,old.occupations,saddle_index=None,freeze_and_release=True,freeze_iterations=150,maximum_iterations=250,orbital_tolerance=1e-10)
    result=replace(result,reference_state=ctx.state,core_electrons=old.core_electrons,spin_reference_fingerprint=ref.fingerprint)
    row=dict(component=i,index=result.orbital_saddle_index,old_index=old.orbital_saddle_index,energy_difference_ev=(result.energy_hartree-old.energy_hartree)*27.211386245988,active=active_orbital_character(old,result,old.reference_state.coordinates_bohr),residual=result.orbital_residual,seconds=time.monotonic()-start)
   except Exception as exc:row=dict(component=i,error=str(exc),seconds=time.monotonic()-start)
   rows.append(row);print(name,json.dumps(row),flush=True)
  (BASE/'results'/name/'freeze_release_probe.json').write_text(json.dumps(rows,indent=2)+'\n')
