"""Parameter-free spectral transport of the ground-state DFT Hamiltonian.

Development only (HNO and uracil). Replaces a raw projected DFT Fock, which does
not preserve DFT eigenvalues, by a Hermitian operator on separately matched
occupied and virtual subspaces. The polar map preserves the selected DFT
spectrum and the native occupied projector. This is a variational one-body
functional, not a state-energy offset. No excitation TBE enters construction.
Both native and complete-response controls are reported, without selecting a
molecule-dependent profile or lambda. All parameters are frozen at R0.
"""
from remote_bootstrap import HOST,BASE
from pathlib import Path
import sys,json,numpy as np
from scipy.linalg import solve_sylvester,polar
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import plan,state_from,write,digest,key
import full_response as full
from pyscf import lib
lib.num_threads(2)
ROOT=BASE/'results/spectral_response';ROOT.mkdir(exist_ok=True)
CASES=[c for c in plan() if c['name'] in ['nitroxyl','uracil']]
LABELS=['spectral_native_response','spectral_full_response']
protocol=dict(host=HOST,script_sha256=digest(__file__),scope='development only; no holdout reuse for candidate selection',cases=['nitroxyl','uracil'],variants=LABELS,
 orbital_mapping='maximum squared overlap bijection per occupation sector, excluding DFT cores; symmetric polar transport',minimum_subspace_singular_value=.5,
 energy_offset=0.,lambda_exchange=.9,reference_orbital_gap_scale=1.,excited_reference_energies_used_in_model=False,nuclear_derivatives=False,
 gate='report all states; no promotion without balanced npi and pipi improvement and subsequent unused-molecule validation')
p=ROOT/'protocol.json'
if p.exists():assert json.loads(p.read_text())==protocol
else:write(p,protocol)

def calibrate(case,s,ctx,label):
 folder=ROOT/label/case['name'];folder.mkdir(parents=True,exist_ok=True)
 source=BASE/'results/full_response'/case['name']/'model.npz'
 request=key(dict(protocol=digest(p),source_model=digest(source),reference=digest(Path(case['reference'])/'reference.npz'),variant=label))
 final=folder/'model.npz'
 if final.exists():
  with np.load(final) as f:assert str(f['request_hash'])==request;return {k:f[k].copy() for k in f.files if k!='request_hash'},request
 with np.load(source) as f:data={k:f[k].copy() for k in f.files if k!='request_hash'}
 old=Path('/home/enzo/MOUSE-roks-repair-20261030/results/spin_kernel')/case['name']/'kernel.npz'
 with np.load(old) as f:lift=f['embedding']
 with np.load(Path(case['reference'])/'reference.npz') as f:cd=f['coefficients'];sd=f['overlap'];occ=f['occupations'];ed=f['orbital_energies_hartree']
 overlaps=s.coefficients.T@lift.T@sd@cd
 occupied=np.flatnonzero(s.alpha_occupations==1);virtual=np.flatnonzero(s.alpha_occupations==0)
 ncore=int(np.sum(s.atomic_numbers-np.asarray(ctx.calculator._valence))/2)
 h=np.zeros((len(s.overlap),len(s.overlap)));mapping=[]
 for ids,refs in [(occupied,np.flatnonzero(occ==2)[ncore:]),(virtual,np.flatnonzero(occ==0))]:
  cross=overlaps[np.ix_(ids,refs)];ii,jj=linear_sum_assignment(-cross**2);assert np.array_equal(ii,np.arange(len(ids)))
  selected=refs[jj];cross=overlaps[np.ix_(ids,selected)];sv=np.linalg.svd(cross,compute_uv=False)
  if np.min(sv)<.5:raise ValueError('unqualified compact DFT subspace: min singular value '+str(np.min(sv)))
  rotation,_=polar(cross);h[np.ix_(ids,ids)]=(rotation*ed[selected])@rotation.T
  mapping.append(dict(source_indices=ids,reference_indices=selected,singular_values=sv,individual_overlaps=abs(np.diag(cross)),eigenvalue_error=float(np.max(abs(np.linalg.eigvalsh(h[np.ix_(ids,ids)])-np.sort(ed[selected]))))))
 potential=h-np.diag(s.orbital_energies_hartree);potential-=np.eye(len(h))*np.trace(potential)/len(h)
 vao=s.overlap@s.coefficients@potential@s.coefficients.T@s.overlap
 a=data['operators'];dual=np.array([2*solve_sylvester(s.overlap,s.overlap,x) for x in a])
 data['linear']=np.einsum('aij,ij->a',dual,vao)
 if label=='spectral_native_response':
  with np.load(BASE/'results/response_ablation/native_control'/case['name']/'model.npz') as f:data['spin']=f['spin']
  data['charge']=np.zeros_like(data['charge'])
 np.savez_compressed(final,request_hash=request,**data)
 write(folder/'mapping.json',dict(host=HOST,request_hash=request,mapping=mapping,occupied_virtual_coupling_norm=float(np.linalg.norm(h[np.ix_(occupied,virtual)])),
  source_model_sha256=digest(source),reference_sha256=digest(Path(case['reference'])/'reference.npz'),scalar_energy_offset=0.,reference_gap_scale=1.,lambda_exchange=.9,
  notes='one-body operator in native AO metric; constant uniform potential gauge removed; same functional for ground and excited states'))
 return data,request
with threadpool_limits(limits=2,user_api='blas'):
 for case in CASES:
  for label in LABELS:
   full.ROOT=ROOT/label;full.ROOT.mkdir(exist_ok=True)
   full.calibrate=lambda case,s,ctx,label=label:calibrate(case,s,ctx,label)
   full.run(case)
rows=[]
for case in CASES:
 for target in case['targets']:
  row=dict(molecule=case['name'],state=target['id'],tbe_ev=target['tbe_ev'],variants={})
  for label in LABELS:row['variants'][label]=json.loads((ROOT/label/case['name']/target['id']/'result.json').read_text())
  rows.append(row)
write(ROOT/'summary.json',dict(host=HOST,protocol_sha256=digest(p),scope=protocol['scope'],rows=rows))
print('SPECTRAL COMPLETE',flush=True)
