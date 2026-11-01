"""Constrained static screening with an explicit orthogonal DFT bath.

Native valence orbitals and the eliminated Gaussian bath are orthogonalized
as separate sectors. Mixed Coulomb vertices, rather than projected-density
weights on active transitions, include genuine bath polarization without
screening an active transition by its own missing spectral weight.
"""
from prepare_fresh import *
from scipy.linalg import null_space
from pyscf import df,lib
from matrix_gfn2.nosi_surface import PreparedNOSIModel,load_nosi_result,save_nosi_result
import shutil
sys.path.insert(0,str(Q));from screening import sector

def run_screen(cohort,name,auxbasis='weigend+etb',bath_strength=1.,reference_parent='downfold_reference'):
 assert 0<=bath_strength<=1
 parent=HERE/reference_parent/cohort/name;tag='' if reference_parent=='downfold_reference' else '_'+reference_parent;cache_root=HERE/('disentangled_bath'+tag if auxbasis=='weigend+etb' else 'disentangled_bath_auxcheck'+tag)/cohort/name;cache_root.mkdir(parents=True,exist_ok=True);out=(cache_root if bath_strength==1 else HERE/f'disentangled_bath{tag}_eta_{bath_strength:.6f}'/cohort/name);out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():return
 model=PreparedNOSIModel(parent/'model.json');r=load_nosi_result(parent/'checkpoint.npz',model);case=prepare(cohort,name);s=case['s'];no=case['no'];n=len(s.overlap);sr=model._anchor_sqrt;si=np.linalg.inv(sr);ctx=prepare_gfn2_deltascf(case['calc'],case['xyz'],orbital_reference=model.anchor,spin_reference=model.response);p=r.ground_coefficients[:,:no]@r.ground_coefficients[:,:no].T;fg,_=ctx.fock_and_energy(np.array([p,p]));en,canonical=eigh(fg[0],s.overlap);y=sr@canonical
 # The excluded active span is exactly the frozen retained CI chart.
 frozen=model.arrays['auxiliary_frame'];hs=[np.eye(no)[:,i] for i in range(max(0,no-5),no)];ps=[np.eye(n-no)[:,i] for i in range(min(5,n-no))]
 for c in r.state_coefficients:
  hs.append(frozen[:,:no].T@sr@c[:,0]);ps.append(frozen[:,no:].T@sr@c[:,no])
 oh=sector(hs,no);ov=sector(ps,n-no);active_h=frozen[:,:no]@oh;active_p=frozen[:,no:]@ov;wh=np.sum((active_h.T@y[:,:no])**2,axis=0);wp=np.sum((active_p.T@y[:,no:])**2,axis=0);fraction=np.clip(1-wh[:,None]*wp[None,:],0,1)
 with np.load(CAM/cohort/name/'CAM_full_reference/reference.npz') as z:cd=z['coefficients'].copy();ed=z['orbital_energies'].copy();od=z['occupations'].copy()
 nc=np.count_nonzero(od==2)-no;projection=cd.T@case['sd']@case['lift']@si;core=cd.T@case['sd']@case['cd'][:,:nc];assert np.max(abs(projection.T@projection-np.eye(n)))<1e-8;assert np.max(abs(core.T@projection),initial=0)<1e-8
 bath=null_space(np.column_stack([projection,core]).T);eb,ub=eigh(bath.T@(ed[:,None]*bath));bath=cd@bath@ub
 # Align the arbitrary one-body gauge once, using occupied centroids.
 gauge=float(np.mean(en[:no])-np.mean(ed[nc:nc+no]));eb=eb+gauge;native_gap=en[no:][None,:]-en[:no,None];bath_gap=eb[None,:]-en[:no,None];assert min(native_gap.min(),bath_gap.min())>1e-7
 vertices=cache_root/'vertices.npz'
 if vertices.exists():
  with np.load(vertices) as z:assert str(z['parent'])==model.fingerprint;lp=z['lp'].copy();ln=z['ln'].copy();lb=z['lb'].copy()
 else:
  cache=cache_root/'cderi.npz'
  if not cache.exists() and reference_parent!='downfold_reference':
   reusable=HERE/'disentangled_bath'/cohort/name/'cderi.npz'
   if reusable.exists():shutil.copy2(reusable,cache)
  if cache.exists():
   with np.load(cache) as z:three=z['three'].copy()
  else:three=df.incore.cholesky_eri(case['mol'],auxbasis=auxbasis,max_memory=3000);np.savez_compressed(cache,three=three)
  phi=case['lift']@si;co=case['lift']@canonical[:,:no];cv=case['lift']@canonical[:,no:];virtual=np.column_stack([cv,bath]);lp=np.empty((len(three),n*n));ln=np.empty((len(three),no*(n-no)));lb=np.empty((len(three),no*len(eb)))
  for lo in range(0,len(three),16):
   hi=min(lo+16,len(three));block=lib.unpack_tril(three[lo:hi]);lp[lo:hi]=np.einsum('Lmn,mp,nq->Lpq',block,phi,phi,optimize=True).reshape(hi-lo,-1);transition=np.einsum('mi,Lmn,nj->Lij',co,block,virtual,optimize=True);ln[lo:hi]=transition[:,:,:n-no].reshape(hi-lo,-1);lb[lo:hi]=transition[:,:,n-no:].reshape(hi-lo,-1)
  np.savez_compressed(vertices,lp=lp,ln=ln,lb=lb,parent=model.fingerprint)
 bn=2*ln*np.sqrt(fraction.ravel()/native_gap.ravel())[None,:];bb=2*lb/np.sqrt(bath_gap.ravel())[None,:];dielectric=np.eye(len(lp))+bn@bn.T+bath_strength*(bb@bb.T);screen=lp.T@np.linalg.solve(dielectric,lp);screen=.5*(screen+screen.T);w=screen.reshape((n,)*4);pg=2*sr@p@sr;h=si@fg[0]@si-np.einsum('pqrs,rs->pq',w,pg)+.5*np.einsum('prqs,rs->pq',w,pg)
 for filename in ['template.npz','response.npz']:shutil.copy2(parent/filename,out/filename)
 arrays=dict(model.arrays);arrays.update(h=h,eri=w,ground=r.ground_coefficients,states=r.state_coefficients,multipliers=r.multipliers,reference_ci=r.ci_coefficients);np.savez_compressed(out/'arrays.npz',**arrays);manifest=dict(model.manifest);manifest['sha256']={f:sha(out/f) for f in ['template.npz','response.npz','arrays.npz']};manifest['provenance']=dict(manifest['provenance'],polarization_chart='native_valence_plus_orthogonal_full_DFT_bath',native_active_space='frozen retained CI chart plus retained target spans',auxbasis=auxbasis,parent_model=model.fingerprint,bath_polarization_strength=bath_strength,bath_strength_fitted=bath_strength!=1);write(out/'model.json',manifest)
 final=PreparedNOSIModel(out/'model.json');result=final.evaluate(case['calc'],case['xyz'],properties=('energy',));save_nosi_result(out/'checkpoint.npz',result)
 bare=np.einsum('pqrs,pi,qj,rk,sl->ijkl',case['eri'],si,si,si,si,optimize=True).reshape(n*n,n*n);df_error=float(np.max(abs(lp.T@lp-bare)));report=dict(bath_polarization_strength=bath_strength,bath_strength_fitted=bath_strength!=1,cohort=cohort,molecule=name,states=[dict(id=v.state_id,gap_ev=v.excitation_hartree*EV,overlap=v.overlap,margin=v.margin) for v in result.states],model=final.fingerprint,weights=[.5,.5],lambda_exchange=.9,offset_ev=0,reference_excitation_energies_read=False,auxbasis=auxbasis,auxiliary_functions=len(lp),bath_virtuals=len(eb),native_environment_transitions=int(np.count_nonzero(fraction>1e-10)),bath_transitions=lb.shape[1],gauge_alignment_hartree=gauge,minimum_bath_gap_hartree=float(bath_gap.min()),bare_DF_integral_max_error_hartree=df_error,minimum_dielectric_eigenvalue=float(np.linalg.eigvalsh(dielectric).min()),screened_coulomb_trace=float(np.trace(screen)))
 write(out/'result.json',report);print(name,report['states'],'DF max',df_error,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('cohort');p.add_argument('name');p.add_argument('--auxbasis',default='weigend+etb');p.add_argument('--bath-strength',type=float,default=1.);p.add_argument('--reference-parent',default='downfold_reference');a=p.parse_args()
 with threadpool_limits(2):run_screen(a.cohort,a.name,a.auxbasis,a.bath_strength,a.reference_parent)
