"""Independent integration-grid convergence of the new frozen-energy block."""
from remote_bootstrap import HOST
from pathlib import Path
import json,hashlib,time,numpy as np
from pyscf import dft,scf,lib
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'frozen_relaxation';lib.num_threads(2)
plan=json.loads((OUT/'plan.json').read_text());records=[]
for name in ['nitroxyl','uracil']:
 case=next(x for x in plan['cases'] if x['name']==name);folder=Path(case['reference']);raw=json.loads((OUT/name/'result.json').read_text())
 cache=OUT/name/'grid_level4.json';refhash=hashlib.sha256((folder/'reference.npz').read_bytes()).hexdigest()
 request=hashlib.sha256(json.dumps(dict(reference=refhash,grid_level=4,xc='PBE0',small_rho_cutoff=0.,schema='grid-check-v1'),sort_keys=True).encode()).hexdigest()
 if cache.exists():
  saved=json.loads(cache.read_text());assert saved['request_hash']==request;records+=saved['rows'];continue
 with np.load(folder/'reference.npz') as f:c=f['coefficients'];occ=f['occupations'];s=f['overlap']
 mol=scf.chkfile.load_mol(str(folder/'backend.chk'));mol.max_memory=6000;mol.verbose=0
 mf=dft.UKS(mol);mf.xc='PBE0';mf.grids.level=4;mf.grids.verbose=0;mf.small_rho_cutoff=0.;mf.max_memory=6000;mf.verbose=0;mf.grids.build(with_non0tab=True)
 def dm(o):return np.array([(c*x)@c.T for x in o])
 def energy(p):
  assert abs(np.einsum('smn,nm->',p,s)-sum(occ))<1e-7
  return float(mf.energy_tot(dm=p))
 eg=energy(dm([occ/2,occ/2]));rows=[]
 for target in raw['records']:
  if not target['qualified']:continue
  h=target['dft_hole'];p=target['dft_particle'];om=np.array([occ/2,occ/2]);om[0,h]=0;om[0,p]=1;ot=np.array([np.maximum(*om),np.minimum(*om)])
  frozen=(2*energy(dm(om))-energy(dm(ot))-eg)*27.211386245988
  row=dict(molecule=name,state=target['id'],level3_frozen_ev=target['dft_frozen_ev'],level4_frozen_ev=frozen,grid_difference_ev=frozen-target['dft_frozen_ev']);rows.append(row);print(row,flush=True)
 cache.write_text(json.dumps(dict(host=HOST,request_hash=request,grid_points=len(mf.grids.coords),rows=rows),indent=2)+'\n');records+=rows
maximum=max(abs(x['grid_difference_ev']) for x in records)
result=dict(host=HOST,maximum_difference_ev=maximum,tolerance_ev=.005,passed=maximum<.005,rows=records)
(OUT/'grid_validation.json').write_text(json.dumps(result,indent=2)+'\n')
if not result['passed']:raise ValueError('grid dependence exceeds declared 5 meV tolerance')
