"""Same KKT equations, with multipliers eliminated from the trust metric."""
from joint_orthogonal import *

def solve_joint_projected(objects,grounds,point,out):
    out.mkdir(parents=True,exist_ok=True);frame=RelativeActiveFrame(objects);constraints=overlap_frames(objects,grounds)
    target=point[0][:,frame.active].T@frame.s@point[1][:,frame.active]
    checkpoint=out/'checkpoint.npz'
    if checkpoint.exists():
        with np.load(checkpoint) as a:point=[c.copy() for c in a['coefficients']]
    history=json.loads((out/'history.json').read_text()) if (out/'history.json').exists() else []
    def evaluate(point):
        e,g,h,t,rawmap,v=frame.evaluate(point)
        for f in constraints:f.maps=lambda pt:(t,rawmap,v)
        rows=[f.evaluate(point) for f in constraints];values=np.array([r[0] for r in rows]);jac=np.array([r[1] for r in rows]);ch=np.array([r[2] for r in rows])
        mu=np.linalg.lstsq(jac.T,g,rcond=1e-10)[0];res=np.r_[g-mu@jac,values]
        return e,g,h,values,jac,ch,mu,res,rawmap,v
    for unused in range(100):
        e,g,h,values,jac,ch,mu,res,rawmap,v=evaluate(point)
        if np.max(abs(res))<1e-8:break
        u,sv,vt=np.linalg.svd(jac,full_matrices=False);normal=vt[sv>1e-10].T
        lag=h-np.einsum('i,ijk->jk',mu,ch)
        projected=lag-normal@(normal.T@lag)
        derivative=np.vstack([projected,jac]);scale=np.linalg.norm(rawmap,axis=0)
        old_ci=[o.cache['ci'].copy() for o in objects];moved=False
        for radius in [.25,.03]:
            step=residual_trust_step(derivative,res,scale,radius)
            for fraction in [1.,.5,.25,.125,.0625,.03125,.015625]:
                trial=frame.retract(point,fraction*step,rawmap,v);*_,rr,_,_=evaluate(trial)
                guard=all(o.fidelity(c)[1]>=.7 and abs(o.signed_overlap(old,ci,c,o.cache['ci']))>=.9 for o,old,ci,c in zip(objects,point,old_ci,trial))
                if guard and np.linalg.norm(rr)<np.linalg.norm(res):point=trial;moved=True;break
            if moved:break
        history.append(dict(iteration=len(history),energy=e,kkt_max=float(np.max(abs(res))),ground_overlaps=values,
            accepted_step=moved,solver='projected KKT; multipliers excluded from physical trust radius'))
        write(out/'history.json',history);np.savez_compressed(out/'checkpoint.tmp.npz',coefficients=np.array(point),multipliers=mu)
        (out/'checkpoint.tmp.npz').replace(checkpoint)
        if not moved:break
    e,g,h,values,jac,ch,mu,res,rawmap,v=evaluate(point);fidelities=[o.fidelity(c)[1] for o,c in zip(objects,point)]
    drift=float(np.max(abs(point[0][:,frame.active].T@frame.s@point[1][:,frame.active]-target)))
    result=dict(qualified=bool(np.max(abs(res))<1e-8 and min(fidelities)>=.7 and drift<1e-9),energy_hartree=e,
        parent_energies=[o.evaluate(c,False)[0] for o,c in zip(objects,point)],kkt_max=float(np.max(abs(res))),
        ground_overlaps=values,state_fidelities=fidelities,active_frame_drift=drift,iterations=len(history),multipliers=mu,
        source_sha256=sha(__file__),host=HOST,threads=2,nuclear_response_implemented=False)
    write(out/'result.json',result);np.savez_compressed(out/'final.npz',coefficients=np.array(point))
    return result,point

if __name__=='__main__':
    import predict_joint as driver
    driver.solve_joint=solve_joint_projected
    lock=(P/'blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    with threadpool_limits():driver.predict_joint('independent_challenge','Furofuran')
