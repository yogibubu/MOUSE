"""Exact cached/uncached range J/K and nonstationary response equivalence."""
import runpy,socket,json,time
from pathlib import Path
from types import MethodType
import numpy as np
from pyscf import gto,dft,scf
from threadpoolctl import threadpool_limits
assert socket.getfqdn().lower()=='mac-2.homenet.telecomitalia.it'
runpy.run_path('/Users/vincenzobarone/MOUSE-oo-response-20261101/source/qualification/remote_bootstrap.py')
from matrix_pyscf._oo_surface import OrbitalSurface
r=Path(__file__).resolve().parent;rows=[]
with threadpool_limits(2):
 mol=gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',basis='sto-3g',verbose=0)
 archive=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-dft-roks-response-20261102/PBE0_checkpoint.npz')
 with np.load(archive) as z:c=z['coefficients'];occ=z['occupations']
 rng=np.random.default_rng(83);dm=rng.normal(size=(2,mol.nao_nr(),mol.nao_nr()));dm+=dm.transpose(0,2,1)
 for functional in ['CAM-B3LYP','WB97X']:
  mf=dft.UKS(mol);mf.xc=functional;mf.grids.level=1;mf.small_rho_cutoff=0
  cached=OrbitalSurface(mf,c,occ,common=True);native=OrbitalSurface(mf,c,occ,common=True)
  for w,m,o in native.components:m.get_jk=MethodType(type(m).get_jk,m)
  maximum=0.
  for omega in [.3,-.3]:
   for unused in range(2):
    ja,ka=cached.components[0][1].get_jk(dm=dm,omega=omega)
    jb,kb=native.components[0][1].get_jk(dm=dm,omega=omega)
    maximum=max(maximum,float(np.max(abs(ja-jb))),float(np.max(abs(ka-kb))))
  direction=rng.normal(size=cached.dimension);direction*=.03/np.linalg.norm(direction);rotated=cached.rotate(c,direction)
  a=cached.evaluate(rotated);b=native.evaluate(rotated);v=rng.normal(size=cached.dimension)
  errors=dict(energy=abs(a[0]-b[0]),gradient=float(np.max(abs(a[1]-b[1]))),Jacobian_action=float(np.max(abs(a[2](v)-b[2](v)))))
  assert maximum<1e-11 and max(errors.values())<1e-10,(maximum,errors)
  # A zero budget uses the native exact path without allocating range ERIs.
  limited=dft.UKS(mol);limited.xc=functional;limited.max_memory=0
  surface=OrbitalSurface(limited,c,occ,common=True)
  surface.components[0][1].get_jk(dm=dm,omega=.3)
  assert not surface._range_integrals
  rows.append(dict(functional=functional,maximum_JK_error=maximum,errors=errors,cache=cached.statistics,zero_budget_native_fallback=True))
report=dict(host=socket.getfqdn(),cases=rows,integral_screening_or_truncation=False)
(r/'range_cache_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
