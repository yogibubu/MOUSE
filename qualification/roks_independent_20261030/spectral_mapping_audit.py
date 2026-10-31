"""Record why a whole-spectrum transport is not qualified, without lowering gates."""
from remote_bootstrap import BASE,HOST
from pathlib import Path
import sys,numpy as np
from scipy.linalg import qr
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
sys.path.append('/home/enzo/MOUSE-roks-repair-20261030')
from spin_kernel_probe import plan,state_from,write,digest,EV
ROOT=BASE/'results/spectral_response';rows=[]
with threadpool_limits(limits=2,user_api='blas'):
 for case in [c for c in plan() if c['name'] in ['nitroxyl','uracil']]:
  r,_=state_from(case['targets'][0]);s=r.reference_state
  with np.load(Path('/home/enzo/MOUSE-roks-repair-20261030/results/spin_kernel')/case['name']/'kernel.npz') as f:lift=f['embedding']
  with np.load(Path(case['reference'])/'reference.npz') as f:cd=f['coefficients'];sd=f['overlap'];od=f['occupations'];ed=f['orbital_energies_hartree']
  cross=s.coefficients.T@lift.T@sd@cd;nc=len(np.flatnonzero(od==2))-int(np.sum(s.alpha_occupations));sectors=[];mapped={}
  for occ in [1,0]:
   ids=np.flatnonzero(s.alpha_occupations==occ);refs=np.flatnonzero(od==2)[nc:] if occ else np.flatnonzero(od==0)
   a=cross[np.ix_(ids,refs)];_,j=linear_sum_assignment(-a*a);_,_,piv=qr(a,pivoting=True,mode='economic')
   sv=np.linalg.svd(a[:,j],compute_uv=False);svq=np.linalg.svd(a[:,piv[:len(ids)]],compute_uv=False)
   sectors.append(dict(occupation=occ,source_indices=ids,selected_reference_indices=refs[j],singular_values=sv,qr_singular_values=svq,qualified=bool(sv.min()>=.5)))
   for i,k in enumerate(ids):
    score=abs(a[i,j[i]]);margin=score-np.max(np.delete(abs(a[i]),j[i]),initial=0.)
    mapped[int(k)]=dict(reference_index=int(refs[j[i]]),overlap=float(score),margin=float(margin),energy_ev=float(ed[refs[j[i]]]*EV),qualified=bool(score>=.7 and margin>=.05))
  targets=[]
  for t in case['targets']:
   h,p=mapped[t['hole']],mapped[t['particle']]
   targets.append(dict(state=t['id'],hole=h,particle=p,native_gap_ev=float((s.orbital_energies_hartree[t['particle']]-s.orbital_energies_hartree[t['hole']])*EV),dft_gap_ev=p['energy_ev']-h['energy_ev'],qualified=h['qualified'] and p['qualified']))
  rows.append(dict(molecule=case['name'],sectors=sectors,targets=targets,whole_spectrum_qualified=all(v['qualified'] for v in sectors)))
report=dict(host=HOST,script_sha256=digest(__file__),protocol_sha256=digest(ROOT/'protocol.json'),rows=rows,minimum_subspace_singular_value=.5,
 conclusion='whole-spectrum spectral transport rejected before predictions; occupied subspaces adequate but virtual subspaces ill-conditioned; no threshold relaxation',no_model_promoted=True,no_offset=True)
write(ROOT/'mapping_audit.json',report)
for r in rows:print(r['molecule'],r['whole_spectrum_qualified'],r['targets'],flush=True)
