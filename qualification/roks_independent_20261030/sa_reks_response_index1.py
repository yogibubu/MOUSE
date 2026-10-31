"""Research-only SA-REKS(2,2) with analytic orbital/occupation response.

The orbital Hessian eliminates the variational occupation using its Schur
complement. This is an ELECTRONIC Hessian, not a nuclear Hessian. Six integer
spin-density microstates call the unchanged ground-state functional.
No fitted energies, gap scaling, or energy offsets. No SI coupling is claimed.
"""
from reks_probe import *
from matrix_gfn2.roks_orbital_response import orbital_residual_jacobian
from matrix_gfn2.orbital_saddle import generalized_mode_following
from matrix_apoc.deltascf import singlet_csf_overlap
from scipy.linalg import expm

OSS=np.array([0.,0.,1.,1.,-.5,-.5])

class SAObjective:
    def __init__(self,ctx,occ):
        self.ctx=ctx;self.occ=occ
        l,r=np.triu_indices(len(occ[0,0]),1);keep=np.any(occ[:,:,l]!=occ[:,:,r],axis=(0,1))
        self.left,self.right=l[keep],r[keep]
    def retract(self,c,step):
        k=np.zeros_like(c);k[self.left,self.right]=step;k[self.right,self.left]=-step
        return c@expm(k)
    def evaluate(self,c,curvature=True):
        l,r=self.left,self.right;energies=[];gradients=[];hessians=[]
        for occ in self.occ:
            f,e=self.ctx.fock_and_energy(np.array([(c*o)@c.T for o in occ]))
            energies.append(e)
            q=sum((c.T@ff@c)*o[None,:] for ff,o in zip(f,occ));gradients.append(2*(q[l,r]-q[r,l]))
            if curvature:
                state=SimpleNamespace(**vars(self.ctx.state));state.coefficients=c;state.alpha_occupations,state.beta_occupations=occ
                kernel=self.ctx.density_response_kernel(state)
                hessians.append(orbital_residual_jacobian(c,occ,f,kernel,self.ctx.polarization,l,r))
        e=np.array(energies);g=np.array(gradients);x,pps=optimize_occupation(e)
        w=.5*(coefficients(x)+OSS);energy=float(w@e);gradient=w@g;h=None
        if curvature:
            h=np.einsum('a,aij->ij',w,hessians)
            if .5<x<1:
                _,gp,gpp=interpolation(x)
                wp=np.array([1.,-1.,-.5*gp,-.5*gp,.5*gp,.5*gp])
                gx=.5*wp@g;hxx=-.25*(e[2]+e[3]-e[4]-e[5])*gpp
                assert hxx>0
                h-=np.outer(gx,gx)/hxx
            h=.5*(h+h.T)
        self.last=dict(x=x,microstate_energies=e,pps=pps,oss=float(OSS@e))
        return energy,gradient,h

def run(case,profile,target_id):
    target=next(t for t in case['targets'] if t['id']==target_id)
    r0,xyz=state_from(target);s=r0.reference_state;name=case['name']
    folder=BASE/'results'/profile/name;root=BASE/'results/sa_reks_response_index1'/profile/name/target_id;root.mkdir(parents=True,exist_ok=True)
    if (root/'result.json').exists():return
    with np.load(folder/'model.npz') as f:data={k:f[k].copy() for k in f.files};model_request=str(f['request_hash'])
    calc=GFN2Calculator([{1:'H',6:'C',7:'N',8:'O'}[int(z)] for z in s.atomic_numbers],temperature_kelvin=0.)
    ground=SimpleNamespace(converged=True,orbital_state=s,energy_hartree=r0.ground_energy_hartree)
    native=prepare_gfn2_deltascf(calc,xyz,ground_state=ground);ref=VerticalReference(s,native.kernel.partition,data,model_request)
    ctx=prepare_gfn2_deltascf(calc,xyz,ground_state=ground,spin_reference=ref)
    old=json.loads((folder/target_id/'result.json').read_text());warm=load_derivative_point(old['checkpoint'],request_hash=old['request_hash'],coordinates_angstrom=xyz,require_gradient=False)
    occ=occupations(s.alpha_occupations,target['hole'],target['particle']);obj=SAObjective(ctx,occ)
    request=key(dict(script=digest(__file__),occupation_script=digest(BASE/'reks_probe.py'),source=model_request,target=target_id,sa_weights=[.5,.5],delta=DELTA))
    c=warm.common_coefficients.copy();cp=root/'latest.npz'
    if cp.exists():
        with np.load(cp) as f:assert str(f['request_hash'])==request;c=f['coefficients']
    # Independent scalar energy check of analytic gradient and Schur response.
    e,g,h=obj.evaluate(c);rng=np.random.default_rng(317);d=rng.normal(size=g.shape);d/=np.linalg.norm(d);checks=[]
    for step in [2e-4,1e-4]:
        ep=obj.evaluate(obj.retract(c,step*d),curvature=False)[0];em=obj.evaluate(obj.retract(c,-step*d),curvature=False)[0]
        checks.append(dict(step=step,gradient_error=abs((ep-em)/(2*step)-g@d),hessian_error=abs((ep-2*e+em)/step**2-d@h@d)))
    assert checks[-1]['gradient_error']<1e-6 and checks[-1]['hessian_error']<2e-5,checks
    write(root/'response_audit.json',dict(host=HOST,checks=checks,scope='analytic electronic orbital Hessian with relaxed occupations; no nuclear Hessian'))
    def checkpoint(point):
        np.savez_compressed(root/'latest.tmp.npz',request_hash=request,coefficients=point.point);(root/'latest.tmp.npz').replace(cp)
        write(root/'history.json',list(point.history))
    start=time.perf_counter();solved=generalized_mode_following(obj.evaluate,obj.retract,c,saddle_index=1,gradient_tolerance=1e-9,maximum_iterations=100,maximum_step=.15,checkpoint=checkpoint)
    obj.evaluate(solved.point);info=obj.last
    adapted=replace(warm,common_coefficients=solved.point);score=singlet_csf_overlap(warm.to_apoc(),adapted.to_apoc(),s.overlap)[0]
    native_score=singlet_csf_overlap(replace(warm,common_coefficients=s.coefficients).to_apoc(),adapted.to_apoc(),s.overlap)[0]
    gap=(info['oss']-info['pps'])*EV
    row=dict(host=HOST,molecule=name,state=target_id,profile=profile,converged=solved.converged,qualified=bool(solved.converged and score>=.7 and native_score>=.7 and gap>0),
             excitation_ev=gap,tbe_ev=target['tbe_ev'],error_ev=gap-target['tbe_ev'],fon=[2*info['x'],2-2*info['x']],warm_csf_overlap=score,native_csf_overlap=native_score,
             observed_index=solved.observed_index,residual=float(np.max(abs(solved.gradient))),seconds=time.perf_counter()-start,request_hash=request,
             scope='SA-REKS vertical exposed development diagnostic; no state interaction, no nuclear derivatives',energy_offset=0.,delta=DELTA)
    np.savez_compressed(root/'final.npz',coefficients=solved.point,request_hash=request,microstate_energies=info['microstate_energies'])
    write(root/'result.json',row);print('SA REKS',row,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--molecule',required=True);ap.add_argument('--profile',choices=['full_response','qualified_spectral_response'],required=True);ap.add_argument('--state',default='n_pi');args=ap.parse_args()
    lib.num_threads(2)
    with threadpool_limits(limits=2,user_api='blas'):run(next(c for c in cases() if c['name']==args.molecule),args.profile,args.state)
