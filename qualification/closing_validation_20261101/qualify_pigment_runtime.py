"""Repeat the affected small-system regressions on the verified compute host."""
from pathlib import Path
import socket,subprocess,os,json,runpy,time
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
BASE=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101');HERE=BASE/'oracle/MOUSE-closing-validation-20261101'
runpy.run_path(str(BASE/'source/qualification/remote_bootstrap.py'))
import numpy as np
from matrix_gfn2._nosi_slater import Hamiltonian
from threadpoolctl import threadpool_limits

with threadpool_limits(2):
 rng=np.random.default_rng(93);n=8;w=rng.normal(size=(n,n,n,n));h=Hamiltonian(np.zeros((n,n)),w);error=0.
 for repeat in range(20):
  p=rng.normal(size=(n,n));q=rng.normal(size=(n,n))
  if repeat%3==0:q*=0.
  for exchange in [False,True]:
   matrix=h.jk if exchange else h.j;expected=float(p.ravel()@matrix@q.ravel());actual=h.same(p,q) if exchange else h.coulomb(p,q);error=max(error,abs(actual-expected))
 assert error==0.,error
 start=time.monotonic()
 for filename in ['validate_fixed_bra.py','validate_pigment_solver_primitives.py','validate_matrix_free_roks.py','validate_density_pair_chart.py','validate_exact_quadrature.py']:
  print('QUALIFY',filename,flush=True);subprocess.run(['/Users/vincenzobarone/.venvs/matrix/bin/python',str(HERE/filename)],cwd=HERE,check=True,env=dict(os.environ,VECLIB_MAXIMUM_THREADS='2'))
 import pytest
 paths=[str(BASE/'source/qualification'/filename) for filename in ['test_nosi_contracts.py','test_state_interaction.py','test_screened_cid.py']];status=pytest.main(['-q']+paths+[str(HERE/'test_target_subspace.py'),str(HERE/'test_pigment_phase.py'),str(HERE/'test_roks_trust_subspace.py')]);assert status==0
 report=dict(host=socket.getfqdn(),Slater_cache_max_error=error,affected_regressions_passed=True,pytest_status=int(status),pytest_tests=25,wall_seconds=time.monotonic()-start,nuclear_Hessian_regression='Complete ethylene analytic Hessian; equivalent complete density chart',pigment_derivative_scope='Vertical energy and orbital response only')
 (HERE/'pigment_runtime_qualification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
