"""Saddle-preserving KKT prototype with analytic orbital constraint curvature.

No nuclear response of the new constraint is implemented by this prototype.
"""
import orthogonal_diagnostic as driver
from orthogonal_diagnostic import *
from orthogonal_singlet import ground_overlap_hessian

def solve_stationary(context,eri,no,c0,ground,transition,out):
    out.mkdir(parents=True,exist_ok=True)
    obj=MultiCSFObjective(context,eri,no,1,ground,transition)
    left,right=obj.left,obj.right;n=len(c0);c=c0.copy();mu=0.
    history=json.loads((out/'history.json').read_text()) if (out/'history.json').exists() else []
    def evaluate(c,mu):
        e,g,h=obj.evaluate(c,True)
        ov,j=ground_overlap_jet(ground,c,context.state.overlap,obj.occupations[0],left,right)
        return e,g,h,ov,j,np.r_[g-mu*j,ov]
    checkpoint=out/'checkpoint.npz'
    if checkpoint.exists():
        with np.load(checkpoint) as a:c=a['coefficients'].copy();mu=float(a['multiplier'])
    accepted=False
    for iteration in range(100):
        e,g,h,ov,j,res=evaluate(c,mu);fidelity=obj.fidelity(c)[1]
        if np.max(abs(res))<1e-8 and fidelity>=.7:accepted=True;break
        hc=ground_overlap_hessian(ground,c,context.state.overlap,obj.occupations[0],left,right)
        system=np.block([[h-mu*hc,-j[:,None]],[j[None,:],np.zeros((1,1))]])
        step=np.linalg.lstsq(system,-res,rcond=1e-10)[0]
        if np.linalg.norm(step[:-1])>.3:step*=.3/np.linalg.norm(step[:-1])
        old_ci=obj.cache['ci'].copy();moved=False
        for fraction in [1.,.5,.25,.125,.0625,.03125,.015625]:
            k=np.zeros((n,n));k[left,right]=fraction*step[:-1];k[right,left]=-k[left,right]
            trial=c@expm(k);new_mu=mu+fraction*step[-1]
            *_,new_res=evaluate(trial,new_mu)
            guard=obj.fidelity(trial)[1]>=.7 and abs(obj.signed_overlap(c,old_ci,trial,obj.cache['ci']))>=.9
            if guard and np.linalg.norm(new_res)<np.linalg.norm(res):
                c=trial;mu=new_mu;moved=True;break
        history.append(dict(iteration=iteration,energy=e,overlap=ov,residual_max=float(np.max(abs(res))),fidelity=fidelity,accepted_step=moved))
        write(out/'history.json',history)
        np.savez_compressed(checkpoint,coefficients=c,multiplier=mu)
        if not moved:break
    e,g,h,ov,j,res=evaluate(c,mu);fidelity=obj.fidelity(c)[1]
    accepted=bool(np.max(abs(res))<1e-8 and fidelity>=.7)
    result=dict(success=accepted,energy_hartree=e,ground_overlap_squared=ov**2,fidelity=fidelity,
        kkt_max=float(np.max(abs(res))),multiplier=mu,qualified=accepted,iterations=len(history),
        solver='KKT stationary point, residual merit, state-overlap guarded',
        constraint_orbital_curvature='analytic, singular-safe determinant second cofactors and exponential retraction',
        source_sha256=sha(__file__),overlap_source_sha256=sha(Path(__file__).parent/'orthogonal_singlet.py'),
        host=HOST,threads=2,checkpoint_resumed=checkpoint.exists())
    np.savez_compressed(out/'final.npz',coefficients=c)
    write(out/'result.json',result);return result

if __name__=='__main__':
    driver.solve=solve_stationary;driver.DEST=DEST/'stationary_analytic'
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():
        for cohort,name,state in [('independent','Criegee_Intermediate','I001'),('independent','Criegee_Intermediate','I002'),('independent','Benzoxadiazole','I004'),('independent','Hexatriene','I005'),('independent_challenge','Furofuran','C001'),('independent_challenge','Furofuran','C002')]:
            driver.run(cohort,name,state)
            print(name,state,'stationary orthogonal pass complete',flush=True)
