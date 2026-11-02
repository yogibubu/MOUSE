"""Replay the archived state through the installed source public API."""
import socket,runpy,json
from pathlib import Path
import numpy as np
from pyscf import gto,dft
from threadpoolctl import threadpool_limits
assert socket.getfqdn().lower()=='mac-2.homenet.telecomitalia.it'
runpy.run_path('/Users/vincenzobarone/MOUSE-oo-response-20261101/source/qualification/remote_bootstrap.py')
from matrix_pyscf import optimize_deltascf
root=Path(__file__).resolve().parent
with threadpool_limits(2):
 mol=gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',basis='sto-3g',verbose=0)
 mf=dft.UKS(mol);mf.xc='PBE0';mf.grids.level=1
 with np.load(root/'PBE0_checkpoint.npz') as z:c=z['coefficients'];occ=z['occupations']
 result=optimize_deltascf(mf,c,occ,restart=root/'PBE0_checkpoint.npz',include_gradient=True,orbital_tolerance=1e-8)
 expected=json.loads((root/'PBE0_validation.json').read_text())
 assert result.iterations==0 and abs(result.energy_hartree-expected['energy_hartree'])<1e-11
 for options in [dict(reference_coefficients=c.astype(complex)),dict(maximum_iterations=0)]:
  try:optimize_deltascf(mf,c,occ,**options)
  except ValueError:pass
  else:raise AssertionError('unsupported input accepted')
 report=dict(public_API=optimize_deltascf.__module__,restart_orbital_iterations=result.iterations,
 energy_error_hartree=abs(result.energy_hartree-expected['energy_hartree']),
 complex_reference_rejected=True,zero_iteration_limit_rejected=True,
 ground_SCF_calls=0,TDA_calls=0,host=socket.getfqdn())
 (root/'public_API_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
