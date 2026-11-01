"""Constrain the full-DFT static response in transition space.

The active particle-hole amplitudes are excluded as complete coherent vectors,
not as diagonal orbital probabilities. Minimize the positive quadratic static
response under A.T X = 0. For D=diag(e_a-e_i), this gives
D^-1 - D^-1 A (A.T D^-1 A)^-1 A.T D^-1.
No excitation reference, fitted scale, or state-specific branch is used.
"""
from prepare_fresh import *
from pyscf import df,lib
from scipy.linalg import null_space
from matrix_gfn2.nosi_surface import PreparedNOSIModel,load_nosi_result,save_nosi_result
import shutil
sys.path.insert(0,str(Q));from screening import sector

def run_screen(cohort,name,mode="constrained_resolvent"):
 parent=HERE/'downfold_reference'/cohort/name;out=HERE/('constrained_transition_screening' if mode=='constrained_resolvent' else 'disentangled_active_screening')/cohort/name;out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():return
 model=PreparedNOSIModel(parent/'model.json');r=load_nosi_result(parent/'checkpoint.npz',model);case=prepare(cohort,name);s=case['s'];no=case['no'];n=len(s.overlap);sr=model._anchor_sqrt;si=np.linalg.inv(sr)
 ctx=prepare_gfn2_deltascf(case['calc'],case['xyz'],orbital_reference=model.anchor,spin_reference=model.response);p=r.ground_coefficients[:,:no]@r.ground_coefficients[:,:no].T;fg,_=ctx.fock_and_energy(np.array([p,p]))
 frozen=model.arrays['auxiliary_frame'];hs=[np.eye(no)[:,i] for i in range(max(0,no-5),no)];ps=[np.eye(n-no)[:,i] for i in range(min(5,n-no))]
 for c in r.state_coefficients:
  hs.append(frozen[:,:no].T@sr@c[:,0]);ps.append(frozen[:,no:].T@sr@c[:,no])
 ah=frozen[:,:no]@sector(hs,no);ap=frozen[:,no:]@sector(ps,n-no)
 with np.load(CAM/cohort/name/'CAM_full_reference/reference.npz') as z:cd=z['coefficients'].copy();ed=z['orbital_energies'].copy();od=z['occupations'].copy()
 nd=np.count_nonzero(od==2);nc=nd-no;co=cd[:,nc:nd];cv=cd[:,nd:];gap=ed[None,nd:]-ed[nc:nd,None];assert gap.min()>0
 # Project and orthonormalize the retained active spaces separately in the
 # full-DFT occupied and virtual sectors. Retain every nonsingular direction.
 def orth(a):
  u,v,_=np.linalg.svd(a,full_matrices=False);assert v.min()>1e-5,('singular active projection',v);return u,v
 uh,sh=orth(co.T@case['sd']@case['lift']@si@ah);uv,sv=orth(cv.T@case['sd']@case['lift']@si@ap)
 a=np.einsum('ih,ap->iahp',uh,uv).reshape(no*cv.shape[1],-1);assert np.max(abs(a.T@a-np.eye(a.shape[1])))<1e-8
 vertices=out/'vertices.npz'
 if mode=='disentangled_active':
  cached=HERE/'constrained_transition_screening'/cohort/name/'vertices.npz'
  if cached.exists() and not vertices.exists():shutil.copy2(cached,vertices)
 if vertices.exists():
  with np.load(vertices) as z:assert str(z['parent'])==model.fingerprint;lp=z['lp'].copy();lt=z['lt'].copy()
 else:
  cache=HERE/'disentangled_bath'/cohort/name/'cderi.npz'
  if cache.exists():
   with np.load(cache) as z:three=z['three'].copy()
  else:
   three=df.incore.cholesky_eri(case['mol'],auxbasis='weigend+etb',max_memory=3000);np.savez_compressed(out/'cderi.npz',three=three)
  phi=case['lift']@si;lp=np.empty((len(three),n*n));lt=np.empty((len(three),no*cv.shape[1]))
  for lo in range(0,len(three),16):
   hi=min(lo+16,len(three));block=lib.unpack_tril(three[lo:hi]);lp[lo:hi]=np.einsum('Lmn,mp,nq->Lpq',block,phi,phi,optimize=True).reshape(hi-lo,-1);lt[lo:hi]=np.einsum('mi,Lmn,nj->Lij',co,block,cv,optimize=True).reshape(hi-lo,-1)
  np.savez_compressed(vertices,lp=lp,lt=lt,parent=model.fingerprint)
 if mode=='constrained_resolvent':
  b=2*lt/np.sqrt(gap.ravel())[None,:];c=a/np.sqrt(gap.ravel())[:,None];bc=b@c;metric=c.T@c;polar=b@b.T-bc@np.linalg.solve(metric,bc.T)
 else:
  # Disentangle active and complementary bands as independent blocks of the
  # full-DFT Fock operator, then exclude only active-to-active transitions.
  def bands(active,energies):
   parts=[];levels=[]
   for sector_basis in [active,null_space(active.T)]:
    if sector_basis.shape[1]:
     ev,rot=eigh(sector_basis.T@(energies[:,None]*sector_basis));parts.append(sector_basis@rot);levels.append(ev)
   return np.column_stack(parts),np.concatenate(levels)
  uo,eo=bands(uh,ed[nc:nd]);ua,ea=bands(uv,ed[nd:]);dg=ea[None,:]-eo[:,None];assert dg.min()>0
  transformed=np.einsum('Lia,ij,ab->Ljb',lt.reshape(len(lp),no,-1),uo,ua,optimize=True);mask=np.ones(dg.shape);mask[:uh.shape[1],:uv.shape[1]]=0
  b=2*transformed.reshape(len(lp),-1)*np.sqrt(mask.ravel()/dg.ravel())[None,:];polar=b@b.T
 polar=.5*(polar+polar.T);pe=np.linalg.eigvalsh(polar);assert pe.min()>-1e-8,pe.min()
 dielectric=np.eye(len(lp))+polar;screen=lp.T@np.linalg.solve(dielectric,lp);screen=.5*(screen+screen.T);w=screen.reshape((n,)*4);pg=2*sr@p@sr;h=si@fg[0]@si-np.einsum('pqrs,rs->pq',w,pg)+.5*np.einsum('prqs,rs->pq',w,pg)
 for filename in ['template.npz','response.npz']:shutil.copy2(parent/filename,out/filename)
 arrays=dict(model.arrays);arrays.update(h=h,eri=w,ground=r.ground_coefficients,states=r.state_coefficients,multipliers=r.multipliers,reference_ci=r.ci_coefficients);np.savez_compressed(out/'arrays.npz',**arrays);manifest=dict(model.manifest);manifest['sha256']={f:sha(out/f) for f in ['template.npz','response.npz','arrays.npz']};manifest['provenance']=dict(manifest['provenance'],polarization_chart=mode,active_constraint='coherent projected frozen retained CI chart',parent_model=model.fingerprint);write(out/'model.json',manifest)
 final=PreparedNOSIModel(out/'model.json');result=final.evaluate(case['calc'],case['xyz'],properties=('energy',));save_nosi_result(out/'checkpoint.npz',result)
 bare=np.einsum('pqrs,pi,qj,rk,sl->ijkl',case['eri'],si,si,si,si,optimize=True).reshape(n*n,n*n);report=dict(cohort=cohort,molecule=name,states=[dict(id=v.state_id,gap_ev=v.excitation_hartree*EV,overlap=v.overlap,margin=v.margin) for v in result.states],model=final.fingerprint,weights=[.5,.5],lambda_exchange=.9,offset_ev=0,reference_excitation_energies_read=False,active_hole_projection_singular_values=sh,active_particle_projection_singular_values=sv,full_transitions=lt.shape[1],excluded_active_directions=a.shape[1],minimum_polarization_eigenvalue=float(pe.min()),bare_DF_integral_max_error_hartree=float(np.max(abs(lp.T@lp-bare))),screened_coulomb_trace=float(np.trace(screen)))
 write(out/'result.json',report);print(name,report['states'],flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('cohort');parser.add_argument('name');parser.add_argument('--mode',choices=['constrained_resolvent','disentangled_active'],default='constrained_resolvent');args=parser.parse_args()
 with threadpool_limits(2):run_screen(args.cohort,args.name,args.mode)
