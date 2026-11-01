"""Validate large-system context against retained native pigment checkpoints."""
from pigment_compact_reference import HERE,load_state
from matrix_gfn2 import GFN2Calculator
from matrix_gfn2.deltascf import prepare_gfn2_deltascf
from matrix_gfn2.calculator import ANGSTROM_TO_BOHR
from types import SimpleNamespace
from threadpoolctl import threadpool_limits
import numpy as np,json,socket
rows=[]
with threadpool_limits(2):
 for name in ['Beta_carotene','Chlorophyllide_a']:
  s=load_state(name);c=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O',12:'Mg'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.);xyz=s.coordinates_bohr/ANGSTROM_TO_BOHR
  try:prepare_gfn2_deltascf(c,xyz,orbital_reference=s)
  except ValueError as e:assert 'allow_large=True' in str(e)
  else:raise AssertionError('large model requires explicit opt-in')
  ctx=prepare_gfn2_deltascf(c,xyz,orbital_reference=s,allow_large=True);p=np.array([(s.coefficients*o)@s.coefficients.T for o in [s.alpha_occupations,s.beta_occupations]]);f,e=ctx.fock_and_energy(p);energy_error=abs(e-s.total_energy_hartree);fock_error=float(np.max(abs(f-s.fock)));assert energy_error<1e-7 and fock_error<1e-8,(name,energy_error,fock_error);print(name,'stationary checkpoint Fock difference',fock_error,flush=True)
  state=SimpleNamespace(**vars(s));kernel=ctx.density_response_kernel(state);rng=np.random.default_rng(19);dp=rng.normal(size=s.overlap.shape);dp=.5*(dp+dp.T);dp/=np.linalg.norm(dp);h=1e-5;fp,ep=ctx.fock_and_energy(p+.5*h*dp[None]);fm,em=ctx.fock_and_energy(p-.5*h*dp[None]);energy_derivative_error=abs((ep-em)/(2*h)-np.sum(f[0]*dp));assert energy_derivative_error<2e-7,(name,'energy/Fock',energy_derivative_error);error=float(np.max(abs((fp-fm)/(2*h)-kernel(dp))));assert error<2e-7
  rows.append(dict(molecule=name,ground_energy_error_hartree=energy_error,preflight_stored_fock_difference_hartree=fock_error,energy_fock_directional_error=energy_derivative_error,response_directional_error=error,checkpoint_reused=True,new_SCC_calls=0,default_size_guard_retained=True))
 report=dict(host=socket.getfqdn(),rows=rows);(HERE/'large_deltascf_certificate.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
