"""Consistent projected-full-DFT constrained static polarization.

The one-body confinement diagnostic is paired with the environment response
of the same complete DFT orbital space. Native GFN2 orbitals, F/Q 50/50,
exchange scale 0.9 and zero offset remain. No excitation label is read.
"""
from prepare_fresh import *
from scipy.linalg import eigh
from matrix_gfn2.nosi_surface import PreparedNOSIModel,load_nosi_result,save_nosi_result
import shutil
sys.path.insert(0,str(Q));from screening import sector

def run_screen(cohort,name):
 parent=HERE/'downfold_reference'/cohort/name;out=HERE/'full_bath_screening'/cohort/name;out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():return
 model=PreparedNOSIModel(parent/'model.json');retained=load_nosi_result(parent/'checkpoint.npz',model);case=prepare(cohort,name);s=case['s'];no=case['no'];n=len(s.overlap);sr=model._anchor_sqrt;si=np.linalg.inv(sr)
 context=prepare_gfn2_deltascf(case['calc'],case['xyz'],orbital_reference=model.anchor,spin_reference=model.response);cg=retained.ground_coefficients;p=cg[:,:no]@cg[:,:no].T;fg,_=context.fock_and_energy(np.array([p,p]));_,canonical=eigh(fg[0],s.overlap);y=sr@canonical
 hs=[np.eye(no)[:,i] for i in range(max(0,no-5),no)];ps=[np.eye(n-no)[:,i] for i in range(min(5,n-no))]
 for c in retained.state_coefficients:
  hs.append(canonical[:,:no].T@s.overlap@c[:,0]);ps.append(canonical[:,no:].T@s.overlap@c[:,no])
 oh=sector(hs,no);ov=sector(ps,n-no);active_h=y[:,:no]@oh;active_p=y[:,no:]@ov
 with np.load(CAM/cohort/name/'CAM_full_reference/reference.npz') as z:cd=z['coefficients'].copy();ed=z['orbital_energies'].copy();od=z['occupations'].copy()
 nc=np.count_nonzero(od==2)-no;occupied=np.arange(nc,nc+no);virtual=np.flatnonzero(od==0);a=si@case['lift'].T@case['sd']@cd;ah=a[:,occupied];ap=a[:,virtual];wh=np.sum((active_h.T@ah)**2,axis=0);wp=np.sum((active_p.T@ap)**2,axis=0);fraction=np.clip(1-wh[:,None]*wp[None,:],0,1)
 ii=np.repeat(np.arange(len(occupied)),len(virtual));aa=np.tile(np.arange(len(virtual)),len(occupied));gap=ed[virtual[aa]]-ed[occupied[ii]];assert gap.min()>1e-7
 t=np.einsum('pi,qi->ipq',ah[:,ii],ap[:,aa]);t+=t.transpose(0,2,1).copy();t=t.reshape(len(ii),n*n).T*np.sqrt(fraction.ravel()/gap)[None,:]
 bare=np.einsum('pqrs,pi,qj,rk,sl->ijkl',case['eri'],si,si,si,si,optimize=True).reshape(n*n,n*n);ev,uv=eigh(bare);keep=ev>1e-11;factor=uv[:,keep]*np.sqrt(ev[keep]);b=factor.T@t;dielectric=np.eye(sum(keep))+b@b.T;screen=factor@np.linalg.solve(dielectric,factor.T);screen=.5*(screen+screen.T);w=screen.reshape((n,)*4)
 pg=2*sr@p@sr;j=np.einsum('pqrs,rs->pq',w,pg);k=np.einsum('prqs,rs->pq',w,pg);h=si@fg[0]@si-j+.5*k
 for filename in ['template.npz','response.npz']:shutil.copy2(parent/filename,out/filename)
 arrays=dict(model.arrays);arrays.update(h=h,eri=w,ground=retained.ground_coefficients,states=retained.state_coefficients,multipliers=retained.multipliers,reference_ci=retained.ci_coefficients);np.savez_compressed(out/'arrays.npz',**arrays);manifest=dict(model.manifest);manifest['sha256']={f:sha(out/f) for f in ['template.npz','response.npz','arrays.npz']};manifest['provenance']=dict(manifest['provenance'],polarization_chart='full_DFT_projected_valence',active_exclusion='1-projected_active_hole_probability*projected_active_particle_probability',parent_model=model.fingerprint,one_shot_screening=True);write(out/'model.json',manifest)
 final=PreparedNOSIModel(out/'model.json');r=final.evaluate(case['calc'],case['xyz'],properties=('energy',));save_nosi_result(out/'checkpoint.npz',r)
 report=dict(cohort=cohort,molecule=name,states=[dict(id=v.state_id,gap_ev=v.excitation_hartree*EV,overlap=v.overlap,margin=v.margin) for v in r.states],model=final.fingerprint,native_orbitals_preserved=True,weights=[.5,.5],lambda_exchange=.9,offset_ev=0,reference_excitation_energies_read=False,full_DFT_environment_transitions=len(ii),native_pair_rank=int(sum(keep)),minimum_gap_hartree=float(gap.min()),active_hole_probabilities=wh,active_particle_probabilities=wp,minimum_dielectric_eigenvalue=float(np.linalg.eigvalsh(dielectric).min()),screened_coulomb_trace=float(np.trace(screen)),bare_coulomb_trace=float(np.trace(bare)))
 write(out/'result.json',report);print(name,report['states'],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('cohort');p.add_argument('name');a=p.parse_args()
 with threadpool_limits(2):run_screen(a.cohort,a.name)
