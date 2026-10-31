"""Analytic orbital response of a nonlocal ensemble-coupling functional.

At the final point E = E_triplet[RSH curvature] + 2(1-xi)(ab|ab).
Both the ground callback and triplet reuse the spin-polarized GFN2 evaluator.
This research adapter keeps the existing DFT first derivative; it is not GX24.
Only electronic orbital derivatives are implemented here, NOT nuclear ones.
"""
from reks_probe import *
from explicit_coupling_probe import get_eri
from edft_rsh_calibration import calibrate,ROOT,XI
from matrix_gfn2.roks_orbital_response import orbital_residual_jacobian
from matrix_gfn2.deltascf import run_deltascf
from matrix_apoc.deltascf import singlet_csf_overlap
from scipy.linalg import expm

def coupling_response(c,eri,hole,particle,left,right,curvature=True):
    """K_ab and its exact real orbital gradient and geodesic Hessian."""
    a,b=c[:,hole],c[:,particle];n=len(c)
    d=.5*(np.outer(a,b)+np.outer(b,a));j=np.einsum('ijkl,kl->ij',eri,d,optimize=True)
    energy=float(np.sum(d*j));ga=2*j@b;gb=2*j@a
    dc=np.zeros((len(left),n,n))
    for z,(i,k) in enumerate(zip(left,right)):dc[z,:,k]=c[:,i];dc[z,:,i]=-c[:,k]
    da,db=dc[:,:,hole],dc[:,:,particle]
    g=da@ga+db@gb
    if not curvature:return energy,g,None
    dd=.5*(da[:,:,None]*b[None,None,:]+a[None,:,None]*db[:,None,:])
    dd+=dd.transpose(0,2,1)
    dj=(eri.reshape(n*n,n*n)@dd.reshape(len(left),-1).T).T.reshape(len(left),n,n)
    dga=2*(dj@b+db@j.T);dgb=2*(dj@a+da@j.T)
    ambient=da@dga.T+db@dgb.T
    # Moving tangent vectors contribute the metric/orthogonality connection.
    # Only the two active columns have a nonzero Cartesian derivative.
    q=np.zeros((n,n));q[:,hole]=c.T@ga;q[:,particle]=c.T@gb
    connection=np.empty_like(ambient)
    for z,(i,k) in enumerate(zip(left,right)):
        dq=np.zeros_like(q);dq[i,:]=-q[k,:];dq[k,:]=q[i,:]
        connection[:,z]=dq[left,right]-dq[right,left]
    h=ambient+connection
    return energy,g,.5*(h+h.T)

class Functional:
    def __init__(self,old,new,occ,triplet,eri,hole,particle):
        self.old,self.new,self.occ,self.triplet,self.eri,self.hole,self.particle=old,new,occ,triplet,eri,hole,particle
        l,r=np.triu_indices(len(eri),1);labels=np.vstack([occ,triplet]);active=np.any(labels[:,l]!=labels[:,r],axis=0)
        self.left,self.right=l[active],r[active];self.theta=0.
    def retract(self,c,step):
        k=np.zeros_like(c);k[self.left,self.right]=step;k[self.right,self.left]=-step
        return c@expm(k)
    def evaluate(self,c,curvature=True):
        t=self.theta;l,r=self.left,self.right;energy=0.;g=np.zeros(len(l));h=np.zeros((len(l),len(l))) if curvature else None
        for weight,ctx,occ in [(2*(1-t),self.old,self.occ),(-(1-t),self.old,self.triplet),(t,self.new,self.triplet)]:
            if abs(weight)<1e-15:continue
            f,e=ctx.fock_and_energy(np.array([(c*o)@c.T for o in occ]));energy+=weight*e
            q=sum((c.T@ff@c)*o[None,:] for ff,o in zip(f,occ));g+=weight*2*(q[l,r]-q[r,l])
            if curvature:
                state=SimpleNamespace(**vars(ctx.state));state.coefficients=c;state.alpha_occupations,state.beta_occupations=occ
                kernel=ctx.density_response_kernel(state)
                h+=weight*orbital_residual_jacobian(c,occ,f,kernel,ctx.polarization,l,r)
        if t:
            ek,gk,hk=coupling_response(c,self.eri,self.hole,self.particle,l,r,curvature)
            scale=2*(1-XI)*t;energy+=scale*ek;g+=scale*gk
            if curvature:h+=scale*hk
        if curvature:h=.5*(h+h.T)
        return float(energy),g,h

CONSISTENT_REFERENCE=False
RESUME_ROOT=None

def run(case,profile,target_id):
    name=case['name'];target=next(t for t in case['targets'] if t['id']==target_id)
    out=ROOT/profile/name/target_id;out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():return
    native_point,xyz=state_from(target);s=native_point.reference_state
    calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
    ground=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=native_point.ground_energy_hartree)
    native=prepare_gfn2_deltascf(calc,xyz,ground_state=ground);data,calreq=calibrate(case,s,native)
    folder=BASE/'results'/profile/name
    with np.load(folder/'model.npz') as f:olddata={k:f[k].copy() for k in f.files};oldreq=str(f['request_hash'])
    if CONSISTENT_REFERENCE:
        data['linear']=data['linear_raw' if profile=='full_response' else 'linear_spectral']
        if profile=='qualified_spectral_response' and not all(data['qualified_mapping'][[target['hole'],target['particle']]]):
            write(out/'result.json',dict(host=HOST,molecule=name,state=target_id,profile=profile,qualified=False,reason='RSH target mapping fails unchanged 0.7/0.05 qualification'));return
    else:data['linear']=olddata['linear']
    modelreq=key(dict(calibration=calreq,profile=profile,first_derivative='consistent RSH' if CONSISTENT_REFERENCE else oldreq))
    np.savez_compressed(out.parent/'model.npz',request_hash=modelreq,**{k:v for k,v in data.items() if k!='request_hash'})
    oldref=VerticalReference(s,native.kernel.partition,olddata,oldreq);newref=VerticalReference(s,native.kernel.partition,data,modelreq)
    oldctx=prepare_gfn2_deltascf(calc,xyz,ground_state=ground,spin_reference=oldref);newctx=prepare_gfn2_deltascf(calc,xyz,ground_state=ground,spin_reference=newref)
    row=json.loads((folder/target_id/'result.json').read_text());warm=load_derivative_point(row['checkpoint'],request_hash=row['request_hash'],coordinates_angstrom=xyz,require_gradient=False)
    if CONSISTENT_REFERENCE:
        with np.load(ROOT/'calibration'/name/'projected_eri.npz') as f:eri=f['eri']
        eri_hash=digest(ROOT/'calibration'/name/'projected_eri.npz')
    else:eri,eri_hash=get_eri(case,s)
    obj=Functional(oldctx,newctx,warm.occupations,warm.triplet_occupations,eri,target['hole'],target['particle'])
    request=key(dict(script=digest(__file__),model=modelreq,source_checkpoint=digest(row['checkpoint']),target=target_id,eri=eri_hash))
    from edft_transition_identity import anchor,transition_overlap
    prepared_anchor=anchor(case,target,s,out,write,digest)
    c=warm.common_coefficients.copy();theta=0.;history=[];cp=out/'latest.npz'
    if not cp.exists() and RESUME_ROOT is not None:
        source_cp=RESUME_ROOT/profile/name/target_id/'latest.npz'
        if source_cp.exists():
            old_request=key(dict(script=digest(BASE/'edft_rsh_response.py'),model=modelreq,source_checkpoint=digest(row['checkpoint']),target=target_id,eri=eri_hash))
            with np.load(source_cp) as f:
                assert str(f['request_hash'])==old_request,'checkpoint model mismatch during tracking-only migration'
                migrated=f['coefficients'].copy();migrated_theta=float(f['theta'])
            source_history=source_cp.parent/'history.json'
            write(out/'history.json',json.loads(source_history.read_text()))
            np.savez_compressed(cp,request_hash=request,coefficients=migrated,theta=migrated_theta)
            write(out/'checkpoint_migration.json',dict(source=str(source_cp),source_sha256=digest(source_cp),old_request_hash=old_request,new_request_hash=request,theta=migrated_theta,energy_functional_changed=False,reason='replace canonical-ground overlap proxy by tracked-excited-CSF and frozen DFT-transition identities'))
    if cp.exists():
        with np.load(cp) as f:assert str(f['request_hash'])==request;c=f['coefficients'];theta=float(f['theta'])
        history=json.loads((out/'history.json').read_text())
    def scores(coeff):
        state=replace(warm,common_coefficients=coeff).to_apoc()
        return [singlet_csf_overlap(warm.to_apoc(),state,s.overlap)[0],transition_overlap(coeff,target['hole'],target['particle'],s,prepared_anchor)]
    def save():
        np.savez_compressed(out/'latest.tmp.npz',request_hash=request,coefficients=c,theta=theta);(out/'latest.tmp.npz').replace(cp)
        write(out/'history.json',history)
    # Scalar energy audit, independent of the derivative formula and solver.
    if not (out/'response_audit.json').exists():
        obj.theta=1.;e,g,h=obj.evaluate(c);rng=np.random.default_rng(9281);d=rng.normal(size=len(g));d/=np.linalg.norm(d);checks=[]
        for step in [2e-4,1e-4]:
            ep=obj.evaluate(obj.retract(c,step*d),False)[0];em=obj.evaluate(obj.retract(c,-step*d),False)[0]
            checks.append(dict(step=step,gradient_error=abs((ep-em)/(2*step)-g@d),hessian_error=abs((ep-2*e+em)/step**2-d@h@d)))
        write(out/'response_audit.json',dict(host=HOST,checks=checks,scope='analytic ELECTRONIC orbital Hessian; not nuclear'))
        assert checks[-1]['gradient_error']<1e-6 and checks[-1]['hessian_error']<2e-5,checks
    start=time.perf_counter();increment=.1;failed=None
    while theta<1-1e-12:
        trial_theta=min(1.,theta+increment);obj.theta=trial_theta;trial=c.copy();converged=False
        for iteration in range(50):
            e,g,h=obj.evaluate(trial);residual=float(np.max(abs(g),initial=0.));values=np.linalg.eigvalsh(h)
            if residual<1e-9:
                converged=bool(min(scores(trial))>=.7 and min(abs(values),default=1)>1e-7);break
            step=np.linalg.lstsq(h,-g,rcond=1e-10)[0];length=np.linalg.norm(step)
            if length>.2:step*=.2/length
            accepted=False
            for fraction in [1.,.5,.25,.125,.0625,.03125]:
                candidate=obj.retract(trial,fraction*step)
                step_overlap=singlet_csf_overlap(replace(warm,common_coefficients=trial).to_apoc(),replace(warm,common_coefficients=candidate).to_apoc(),s.overlap)[0]
                if step_overlap<.7 or min(scores(candidate))<.7:continue
                _,gg,_=obj.evaluate(candidate,False)
                if np.linalg.norm(gg)<np.linalg.norm(g):trial=candidate;accepted=True;break
            if not accepted:break
        attempt=dict(theta=trial_theta,iterations=iteration+1,residual=residual,index=int(np.count_nonzero(values<-1e-7)),csf_overlaps=scores(trial),converged=converged)
        history.append(attempt)
        if converged:c=trial;theta=trial_theta;increment=min(.1,increment*1.5);save();print('EDFT STAGE',name,profile,target_id,attempt,flush=True)
        else:
            increment*=.5;save()
            if increment<.001:failed='continuation could not retain a converged state; no prediction';break
    if failed:
        write(out/'result.json',dict(host=HOST,molecule=name,state=target_id,profile=profile,qualified=False,reason=failed,theta=theta,request_hash=request));return
    # Reuse the converged old ground orbitals as a guess, never reuse its energy.
    with np.load(folder/'tda.npz') as f:ground_coeff=f['ground_coefficients']
    gp=run_deltascf(s.overlap,ground_coeff,np.array([s.alpha_occupations,s.beta_occupations]),newctx.fock_and_energy,
                   selection='aufbau',density_tolerance=1e-10,orbital_tolerance=1e-10,energy_tolerance=1e-12,maximum_iterations=500)
    obj.theta=1.;e,g,h=obj.evaluate(c);gap=(e-gp.energy_hartree)*EV;ss=scores(c);values=np.linalg.eigvalsh(h)
    result=dict(host=HOST,molecule=name,state=target_id,profile=profile,qualified=bool(gap>0 and min(ss)>=.7 and np.max(abs(g))<=1e-9),
                excitation_ev=gap,tbe_ev=target['tbe_ev'],error_ev=gap-target['tbe_ev'],csf_overlaps=ss,orbital_residual=float(np.max(abs(g))),
                electronic_saddle_index=int(np.count_nonzero(values<-1e-7)),seconds=time.perf_counter()-start,request_hash=request,energy_offset=0.,xi=XI,
                label='GFN2 EDFT/RSH response research adapter; not full GX24',nuclear_derivatives_supported=False,
                ground_energy_hartree=gp.energy_hartree,excited_energy_hartree=e,identity_fields=['tracked_excited_csf_overlap','frozen_DFT_transition_overlap'],
                native_canonical_csf_overlap=singlet_csf_overlap(replace(warm,common_coefficients=s.coefficients).to_apoc(),replace(warm,common_coefficients=c).to_apoc(),s.overlap)[0])
    np.savez_compressed(out/'final.npz',coefficients=c,ground_coefficients=gp.coefficients,ground_energy=gp.energy_hartree,excited_energy=e,request_hash=request)
    write(out/'result.json',result);print('EDFT FINAL',result,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--molecule',required=True);ap.add_argument('--profile',choices=['full_response','qualified_spectral_response'],required=True);ap.add_argument('--state',default='n_pi');ap.add_argument('--consistent-reference',action='store_true');args=ap.parse_args()
    if args.consistent_reference:
        from edft_rsh_consistent_calibration import calibrate,ROOT as ORIGINAL_ROOT
        RESUME_ROOT=ORIGINAL_ROOT;ROOT=BASE/'results/edft_rsh_statefollow';ROOT.mkdir(exist_ok=True)
        CONSISTENT_REFERENCE=True
    lib.num_threads(2)
    with threadpool_limits(limits=2,user_api='blas'):run(next(c for c in cases() if c['name']==args.molecule),args.profile,args.state)
