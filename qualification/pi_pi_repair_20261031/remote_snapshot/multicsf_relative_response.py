"""Experimental two-parent active-orbital constraints and analytic response.

Each corresponding active hole/particle overlap squared is held at its anchor
value. Both parents use the same metric-compatible nuclear chart, so these
constraints have identically zero explicit nuclear derivatives. Unit-overlap
singularities require shared-coordinate elimination and are rejected here.
"""
import numpy as np
from scipy.linalg import block_diag,expm


def constraint_jet(objects,coefficients,target):
    sizes=[len(o.left) for o in objects];active=np.r_[np.arange(objects[0].rank),objects[0].no+np.arange(objects[0].rank)]
    assert objects[0].rank==objects[1].rank
    s=objects[0].context.state.overlap
    generators=[];first=[]
    for o,c in zip(objects,coefficients):
        g=np.zeros((len(o.left),len(c),len(c)))
        g[np.arange(len(g)),o.left,o.right]=1
        g[np.arange(len(g)),o.right,o.left]=-1
        generators.append(g);first.append(c@g)
    values=[];jac=[];hess=[];overlaps=[]
    for a,t in zip(active,target):
        x,y=coefficients[0][:,a],coefficients[1][:,a];v=float(x@s@y)
        d=np.r_[first[0][:,:,a]@s@y,first[1][:,:,a]@s@x]
        second=[]
        for side,other in [(0,y),(1,x)]:
            raw=np.einsum('kmi,li->klm',first[side],generators[side][:,:,a],optimize=True)
            second.append(np.einsum('klm,m->kl',.5*(raw+raw.transpose(1,0,2)),s@other))
        cross=first[0][:,:,a]@s@first[1][:,:,a].T
        vv=np.block([[second[0],cross],[cross.T,second[1]]])
        values.append(v*v-t*t);jac.append(2*v*d);hess.append(2*np.outer(d,d)+2*v*vv);overlaps.append(v)
    return np.array(values),np.array(jac),np.array(hess),np.array(overlaps)


def anchor_overlaps(objects,coefficients):
    o=objects[0];active=np.r_[np.arange(o.rank),o.no+np.arange(o.rank)]
    return np.array([coefficients[0][:,a]@o.context.state.overlap@coefficients[1][:,a] for a in active])


def stationary_blocks(objects,coefficients,target,multipliers,weights=(.5,.5)):
    data=[o.evaluate(c,True) for o,c in zip(objects,coefficients)]
    e=sum(w*d[0] for w,d in zip(weights,data));g=np.concatenate([w*d[1] for w,d in zip(weights,data)])
    h=block_diag(*[w*d[2] for w,d in zip(weights,data)])
    q,j,qh,overlaps=constraint_jet(objects,coefficients,target)
    singular=np.linalg.svd(j,compute_uv=False)
    if singular[-1]<1e-8:raise ValueError('active constraint singularity: shared-coordinate elimination required')
    h+=np.einsum('i,ijk->jk',multipliers,qh)
    b=np.block([[h,j.T],[j,np.zeros((len(q),len(q)))]])
    residual=np.r_[g+j.T@multipliers,q]
    return e,residual,b,dict(constraints=q,overlaps=overlaps,minimum_constraint_singular_value=float(singular[-1]))


def polish(objects,coefficients,target,multipliers=None,weights=(.5,.5)):
    """Local KKT Newton continuation; preserve identity and coherent overlap."""
    coefficients=[c.copy() for c in coefficients]
    if multipliers is None:multipliers=np.zeros(len(target))
    sizes=[len(o.left) for o in objects];offset=np.cumsum([0]+sizes)
    for iteration in range(16):
        e,res,b,info=stationary_blocks(objects,coefficients,target,multipliers,weights)
        if np.max(abs(res))<1e-10:return coefficients,multipliers,info
        old_ci=[o.cache['ci'].copy() for o in objects]
        step=np.linalg.solve(b,-res);norm=np.linalg.norm(step[:offset[-1]])
        step*=min(1.,.08/max(norm,1e-300));accepted=False
        for fraction in [1.,.5,.25,.125,.0625]:
            trial=[]
            for i,(o,c) in enumerate(zip(objects,coefficients)):
                k=np.zeros_like(c);k[o.left,o.right]=fraction*step[offset[i]:offset[i+1]];k[o.right,o.left]=-k[o.left,o.right]
                trial.append(c@expm(k))
            lm=multipliers+fraction*step[offset[-1]:]
            _,rr,_,_=stationary_blocks(objects,trial,target,lm,weights)
            guards=all(o.fidelity(c)[0]**2>.5 and o.cache['coefficient_margin']>=.05 and abs(o.signed_overlap(old,ci,c,o.cache['ci']))>=.9 for o,old,ci,c in zip(objects,coefficients,old_ci,trial))
            if guards and np.linalg.norm(rr)<np.linalg.norm(res):coefficients=trial;multipliers=lm;accepted=True;break
        if not accepted:raise RuntimeError('joint constrained continuation blocked; point not accepted')
    raise RuntimeError('joint constrained continuation budget exceeded')


def anchor_nuclear_response(objects,coefficients,target,parent_responses,weights=(.5,.5)):
    """Exact constrained parent-mean Hessian at independently stationary anchor.

    This is the excited total-energy mean only. Ground-reference differences
    and native ground-state restoration are separate terms, not included here.
    """
    e,res,b,info=stationary_blocks(objects,coefficients,target,np.zeros(len(target)),weights)
    if np.max(abs(res))>1e-9:raise ValueError('anchor parents are not independently stationary')
    kr=np.concatenate([w*r['orbital_nuclear_coupling'] for w,r in zip(weights,parent_responses)],axis=0)
    forcing=np.concatenate([kr,np.zeros((len(target),kr.shape[1]))],axis=0)
    response=np.linalg.solve(b,-forcing)
    rr=sum(w*r['frozen_hessian'] for w,r in zip(weights,parent_responses))
    h=rr+forcing.T@response
    gradient=sum(w*r['gradient'] for w,r in zip(weights,parent_responses))
    return dict(energy=e,gradient=gradient,hessian=.5*(h+h.T),bordered_hessian=b,response=response,maximum_response_residual=float(np.max(abs(b@response+forcing))),stationarity_residual=float(np.max(abs(res))),minimum_constraint_singular_value=info['minimum_constraint_singular_value'])
