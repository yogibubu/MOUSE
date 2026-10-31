"""Oracle decomposition of exact two-electron and local response couplings.
Diagnostic matrices are not a proposed production functional.
"""
import socket,os,json,sys
from pathlib import Path
assert socket.getfqdn()=='oracle.sns.it'
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
V=Path('/home/enzo/MOUSE-orbital-performance-20261031');P=Path('/home/enzo/MOUSE-publication-validation-20261030');D=Path('/home/enzo/MOUSE-state-following-20261031/reduced_validation');OUT=V/'pi_pi_repair'
EV=27.211386245988
with threadpool_limits(limits=2):
 rows=[]
 for row in json.loads((OUT/'rank_diagnostic_summary.json').read_text())['rows']:
  name=row['molecule'];sid=row['state'];profiles=[]
  for i,base in enumerate(row['profiles']):
   with np.load(P/'results/full_response'/name/'projected_eri.npz') as a:eri=a['eri'].copy()
   with np.load(D/'tda'/name/str(i)/'spectrum.npz') as a:c=a['coefficients'].copy();matrix=a['matrix'].copy()
   sys.path.insert(0,str(V/'source/qualification'));os.environ['MOUSE_VALIDATION_ROOT']=str(V);import remote_bootstrap
   from matrix_gfn2.deltascf_derivatives import load_derivative_point
   p=D/'prepared'/name/'ground.npz'
   with np.load(p) as a:xyz=a['coordinates_angstrom'].copy();req=str(a['request_hash'])
   s=load_derivative_point(p,request_hash=req,coordinates_angstrom=xyz,require_gradient=False).reference_state
   case=json.loads((D/'prepared'/name/'case.json').read_text());t=next(t for t in case['targets'] if t['id']==sid)
   with np.load(t['transition_file']) as a:amp=a['amplitude'].copy();oi=a['occupied'].copy();vi=a['virtual'].copy()
   no=len(oi);x=c[:,:no].T@s.overlap@s.coefficients[:,oi]@amp@s.coefficients[:,vi].T@s.overlap@c[:,no:];x/=np.linalg.norm(x);u,sigma,vh=np.linalg.svd(x,full_matrices=False)
   count=4;co=c[:,:no]@u[:,:count];cv=c[:,no:]@vh[:count].T
   j=np.einsum('mnpq,mk,nk,pl,ql->kl',eri,co,cv,co,cv,optimize=True)*EV
   k=np.einsum('mnpq,mk,nl,pk,ql->kl',eri,co,co,cv,cv,optimize=True)*EV
   a=np.array(base['TDA_projected_ev']);v_hf=2*j-k;v_hybrid=2*j-.225*k;local=a-v_hybrid
   variants=[]
   for label,coupling in [('TDA',a),('Hx_only_full_J',v_hybrid),('hybrid_fraction_times_HF',.225*v_hf),('full_HF',v_hf)]:
    h=coupling.copy();np.fill_diagonal(h,base['ROKS_diagonal_ev']);energies=[]
    for rank in [1,2,3,4]:
     es,z=eigh(h[:rank,:rank]);weights=abs(sigma[:rank]@z);root=int(np.argmax(weights));energies.append(float(es[root]))
    variants.append(dict(coupling=label,rank_energies_ev=energies))
   profiles.append(dict(profile=i,coulomb_2J_ev=(2*j).tolist(),exchange_K_ev=k.tolist(),local_response_offdiag_ev=float(local[0,1]),tda_offdiag_ev=float(a[0,1]),variants=variants))
  rows.append(dict(molecule=name,state=sid,profiles=profiles));print(name,[(p['profile'],round(p['tda_offdiag_ev'],3),round(p['local_response_offdiag_ev'],3),[(v['coupling'],[round(e,3) for e in v['rank_energies_ev'][:2]]) for v in p['variants']]) for p in profiles],flush=True)
 (OUT/'coupling_diagnostic.json').write_text(json.dumps(dict(scope='diagnostic only, no accuracy-based coupling fitted',rows=rows),indent=2)+'\n')
