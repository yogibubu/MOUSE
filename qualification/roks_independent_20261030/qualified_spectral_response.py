"""Development: variational DFT level transport on reliably mapped orbitals.

The whole virtual space cannot be transported faithfully. Only orbitals with
individually qualified DFT correspondences receive a diagonal one-body shift;
unmapped levels retain native spacings in the occupied-centroid gauge. This
profile is fixed before predictions; no state identity or TBE enters its Fock.
Complete charge/spin response is reused, lambda=.9 once. No state offset.
"""
from remote_bootstrap import BASE,HOST
from pathlib import Path
import sys,json,numpy as np
from scipy.linalg import solve_sylvester
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import plan,state_from,write,key,digest
import full_response as full
from pyscf import lib
lib.num_threads(2)
ROOT=BASE/'results/qualified_spectral_response';ROOT.mkdir(exist_ok=True)
CASES=[c for c in plan() if c['name'] in ['nitroxyl','uracil']]
profile=dict(host=HOST,script_sha256=digest(__file__),scope='development only, HNO and uracil',
 mapping='maximum squared overlap per occupation sector in existing isometric DFT embedding',minimum_individual_overlap=.7,minimum_individual_margin=.05,
 potential='diagonal in native canonical orbitals; mapped DFT levels in occupied-centroid gauge; unqualified native levels unchanged',
 response='complete projected DFT charge and spin already calibrated at native reference',lambda_exchange=.9,gap_scale=1.,state_energy_offset=0.,excitation_energies_used_to_build_model=False,
 gate='all 3 states qualified, total MAE below native and raw-full controls, npi MAE below native, pipi absolute error below both controls, maxabs < 0.5 eV; then require unused-molecule validation',nuclear_derivatives=False)
p=ROOT/'protocol.json'
if p.exists():assert json.loads(p.read_text())==profile
else:write(p,profile)

def calibrate(case,s,ctx):
 out=ROOT/case['name'];out.mkdir(exist_ok=True)
 source=BASE/'results/full_response'/case['name']/'model.npz';rfile=Path(case['reference'])/'reference.npz'
 request=key(dict(profile=digest(p),model=digest(source),reference=digest(rfile)))
 if (out/'model.npz').exists():
  with np.load(out/'model.npz') as f:assert str(f['request_hash'])==request;return {k:f[k].copy() for k in f.files if k!='request_hash'},request
 with np.load(source) as f:data={k:f[k].copy() for k in f.files if k!='request_hash'}
 if (BASE/'results/full_response'/case['name']/'embedding.npz').exists():
  with np.load(BASE/'results/full_response'/case['name']/'embedding.npz') as f:lift=f['embedding']
 else:
  with np.load(Path('/home/enzo/MOUSE-roks-repair-20261030/results/spin_kernel')/case['name']/'kernel.npz') as f:lift=f['embedding']
 with np.load(rfile) as f:cd=f['coefficients'];sd=f['overlap'];od=f['occupations'];ed=f['orbital_energies_hartree']
 cross=s.coefficients.T@lift.T@sd@cd;occupied=np.flatnonzero(s.alpha_occupations==1);nc=len(np.flatnonzero(od==2))-len(occupied)
 selected=np.zeros(len(s.overlap),int);score=np.zeros(len(selected));margin=score.copy()
 for occ in [1,0]:
  ids=np.flatnonzero(s.alpha_occupations==occ);refs=np.flatnonzero(od==2)[nc:] if occ else np.flatnonzero(od==0);a=cross[np.ix_(ids,refs)];_,j=linear_sum_assignment(-a*a)
  for i,k in enumerate(ids):selected[k]=refs[j[i]];score[k]=abs(a[i,j[i]]);margin[k]=score[k]-np.max(np.delete(abs(a[i]),j[i]),initial=0.)
 good=(score>=.7)&(margin>=.05)
 # One energy origin for all orbitals, independent of target/state/TBEs.
 ref_origin=float(np.mean(ed[selected[occupied]]));src_origin=float(np.mean(s.orbital_energies_hartree[occupied]))
 delta=np.zeros(len(selected));delta[good]=(ed[selected[good]]-ref_origin)-(s.orbital_energies_hartree[good]-src_origin)
 levels=s.orbital_energies_hartree+delta
 if np.max(levels[occupied])>=np.min(levels[s.alpha_occupations==0]):raise ValueError('spectral transfer loses the native occupied/virtual gap')
 vao=(s.overlap@s.coefficients*delta)@s.coefficients.T@s.overlap
 a=data['operators'];dual=np.array([2*solve_sylvester(s.overlap,s.overlap,x) for x in a]);data['linear']=np.einsum('aij,ij->a',dual,vao)
 # The one-body operator must exactly reproduce prescribed canonical shifts.
 assert np.max(abs(s.coefficients.T@vao@s.coefficients-np.diag(delta)))<1e-9
 np.savez_compressed(out/'model.npz',request_hash=request,**data)
 write(out/'mapping.json',dict(host=HOST,request_hash=request,reference_indices=selected,overlaps=score,margins=margin,qualified=good,delta_hartree=delta,
  source_centroid_hartree=src_origin,reference_centroid_hartree=ref_origin,energy_offset=0.,note='energy-origin alignment of one-body operator; never an excited-state offset'))
 return data,request
with threadpool_limits(limits=2,user_api='blas'):
 full.ROOT=ROOT;full.calibrate=calibrate
 for case in CASES:full.run(case)
rows=[]
for case in CASES:
 for t in case['targets']:
  r=json.loads((ROOT/case['name']/t['id']/'result.json').read_text());mapping=json.loads((ROOT/case['name']/'mapping.json').read_text())
  r['target_orbital_mapping_qualified']=bool(mapping['qualified'][t['hole']] and mapping['qualified'][t['particle']]);r['qualified']=bool(r.get('qualified') and r['target_orbital_mapping_qualified'])
  rows.append(r)
write(ROOT/'summary.json',dict(host=HOST,protocol_sha256=digest(p),scope=profile['scope'],rows=rows))
print('QUALIFIED SPECTRAL COMPLETE',[(r['molecule'],r['state'],r.get('qualified'),r.get('excitation_ev')) for r in rows],flush=True)
