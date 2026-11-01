"""Analytic Newton--Krylov stationarity of one ground-orthogonal ROKS CSF.

The residual and its action are the complete orbital KKT equations. Neither
orbital rotations nor density response are truncated; finite differences do
not enter the solver. Full dense orbital Jacobians are never allocated.
"""
from types import SimpleNamespace
import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator, gmres, minres
from .roks_orbital_response import roks_residual_action
from ._nosi_orthogonality import ground_overlap_jet, prepare_ground_overlap_action
from ._nosi_refine import normalize


def residual_subspace_step(residual,newton,action,radius):
    """Minimize the linearized residual in a Newton/Cauchy trust subspace.

    The symmetric covariant KKT action gives the gradient of its squared
    residual norm. Only the orbital part is radius constrained; the scalar
    Lagrange multiplier is retained in the same two-dimensional minimization.
    No stationary equation or energy parameter is modified.
    """
    gradient=action(residual)
    u,values,_=np.linalg.svd(np.column_stack([newton,-gradient]),full_matrices=False)
    q=u[:,values>max(values[0]*1e-13,1e-15)]
    if not q.shape[1]:raise RuntimeError('empty residual trust subspace')
    b=np.column_stack([action(v) for v in q.T]);h=b.T@b;rhs=-b.T@residual;metric=q[:-1].T@q[:-1]
    def candidate(shift):return np.linalg.lstsq(h+shift*metric,rhs,rcond=1e-14)[0]
    y=candidate(0.);shift=0.
    if np.linalg.norm(q[:-1]@y)>radius:
        lo=0.;hi=max(float(np.linalg.norm(h)),1e-12)
        for unused in range(80):
            if np.linalg.norm(q[:-1]@candidate(hi))<=radius:break
            hi*=2
        else:raise RuntimeError('unbounded orbital trust subproblem')
        for unused in range(60):
            mid=.5*(lo+hi)
            if np.linalg.norm(q[:-1]@candidate(mid))>radius:lo=mid
            else:hi=mid
        shift=hi;y=candidate(shift)
    step=q@y
    return step,dict(trust_subspace_rank=q.shape[1],trust_regularization=shift,
                     predicted_relative_residual=float(np.linalg.norm(residual+b@y)/np.linalg.norm(residual)))


def solve(context, coefficients, ground, transition, multiplier=0., *,
          tolerance=1e-10, maximum_iterations=80, maximum_step=.2, checkpoint=None, linearization_context=None,
          local_state_following=False, krylov='gmres',exact_refinement=False,residual_trust_region=False):
    c=normalize(coefficients,context.state.overlap);n=len(c);no=context.state.electron_count//2
    occ=np.zeros((2,n));occ[:,:no]=1;occ[0,0]=0;occ[0,no]=1
    triplet=np.array([np.maximum(*occ),np.minimum(*occ)])
    left,right=np.triu_indices(n,1);labels=np.vstack([occ,triplet]);keep=np.any(labels[:,left]!=labels[:,right],axis=0);left,right=left[keep],right[keep]
    linear=context if linearization_context is None else linearization_context
    mu=float(multiplier);history=[];s=context.state.overlap;radius=maximum_step
    reference=np.asarray(transition)
    if reference.shape==(no,n-no):reference=reference[None]
    if reference.ndim!=3 or reference.shape[1:]!=(no,n-no):raise ValueError('invalid retained ROKS transition subspace')
    vectors=reference.reshape(len(reference),-1).T
    u,values,_=np.linalg.svd(vectors,full_matrices=False)
    if not len(values) or values[0]<1e-12:raise ValueError('empty retained ROKS transition subspace')
    reference=u[:,values>1e-12*values[0]]
    def csf_overlap(first,second):
        ia=list(range(1,no))+[0];ib=list(range(1,no))+[no]
        aa=first[:,ia];ab=first[:,ib];ba=second[:,ia];bb=second[:,ib]
        return float(np.linalg.det(aa.T@s@ba)*np.linalg.det(ab.T@s@bb)+np.linalg.det(aa.T@s@bb)*np.linalg.det(ab.T@s@ba))
    def evaluate(c,mu):
        components=[];energy=0.;g=np.zeros(len(left))
        for weight,o in [(2.,occ),(-1.,triplet)]:
            p=np.array([(c*x)@c.T for x in o]);f,e=context.fock_and_energy(p)
            energy+=weight*e;q=sum((c.T@ff@c)*x[None,:] for ff,x in zip(f,o));g+=2*weight*(q[left,right]-q[right,left]);components.append((weight,o,f))
        overlap,j=ground_overlap_jet(ground,c,s,occ,left,right)
        transition_now=np.outer(ground[:,:no].T@s@c[:,0],ground[:,no:].T@s@c[:,no]);norm=np.linalg.norm(transition_now)
        fidelity=np.linalg.norm(reference.T@transition_now.ravel())/max(norm,1e-300)
        return float(energy),np.r_[g-mu*j,-overlap],components,j,float(fidelity),float(overlap)
    for iteration in range(maximum_iterations+1):
        energy,residual,components,j,fidelity,overlap=evaluate(c,mu)
        error=float(np.max(abs(residual)));row=dict(iteration=iteration,energy_hartree=energy,residual=error,residual_l2=float(np.linalg.norm(residual)),ground_overlap=overlap,fidelity=fidelity,multiplier=mu);history.append(row)
        if checkpoint is not None:checkpoint(c,mu,row)
        if error<tolerance:return c,mu,dict(converged=True,history=history,energy_hartree=energy,residual=error,ground_overlap=overlap,fidelity=fidelity,retained_transition_rank=reference.shape[1],state_following='accepted-step coherent CSF overlap' if local_state_following else 'anchor transition subspace',krylov=krylov,complete_orbital_rotations=len(left),dense_orbital_Jacobian_allocated=False,full_functional_residual=True,auxiliary_linearization=linearization_context is not None)
        if iteration==maximum_iterations:break
        state=SimpleNamespace(**vars(context.state));state.coefficients=c;state.alpha_occupations,state.beta_occupations=occ
        kernel=linear.density_response_kernel(state)
        constraint=prepare_ground_overlap_action(ground,c,s,occ,left,right) if mu else None
        def action(vector):
            x=vector[:-1];result=roks_residual_action(c,components,kernel,linear.polarization,left,right,x)
            if constraint is not None:result-=mu*constraint(x)[1]
            return np.r_[result-vector[-1]*j,-j@x]
        diagonal=np.zeros(len(left))
        for weight,o,f in components:
            for occupation,ff in zip(o,f):
                levels=np.diag(c.T@ff@c);diagonal+=2*weight*(occupation[right]-occupation[left])*(levels[left]-levels[right])
        # A positive block diagonal preconditioner affects convergence only.
        scale=np.maximum(abs(diagonal),.02);schur=max(float(np.sum(j*j/scale)),1e-4)
        preconditioner=LinearOperator((len(residual),)*2,matvec=lambda x:np.r_[x[:-1]/scale,x[-1]/schur],dtype=float)
        operator=LinearOperator((len(residual),)*2,matvec=action,dtype=float)
        linear_iterations=[0]
        def counted(value):linear_iterations[0]+=1
        rtol=min(.03,max(1e-4,error**.5))
        if krylov=='gmres':step,info=gmres(operator,-residual,M=preconditioner,rtol=rtol,atol=1e-14,restart=40,maxiter=15,callback=counted,callback_type='pr_norm')
        elif krylov=='minres':
            step,info=minres(operator,-residual,M=preconditioner,rtol=min(1e-7,rtol*1e-3),maxiter=600,callback=counted)
            # MINRES uses a backward-error stop, not the relative residual
            # required by inexact Newton. Verify that residual explicitly.
            if np.linalg.norm(action(step)+residual)>rtol*np.linalg.norm(residual):
                step,info=gmres(operator,-residual,x0=step,M=preconditioner,rtol=rtol,atol=1e-14,restart=40,maxiter=15,callback=counted,callback_type='pr_norm')
        else:raise ValueError('unsupported orbital Krylov solver')
        row['Krylov_iterations']=linear_iterations[0];row['Krylov_info']=int(info)
        row['linear_relative_residual']=float(np.linalg.norm(action(step)+residual)/np.linalg.norm(residual))
        if exact_refinement and linearization_context is not None:
            exact_kernel=context.density_response_kernel(state)
            def exact_action(vector):
                x=vector[:-1];result=roks_residual_action(c,components,exact_kernel,context.polarization,left,right,x)
                if constraint is not None:result-=mu*constraint(x)[1]
                return np.r_[result-vector[-1]*j,-j@x]
            # Flexible right-preconditioned GMRES: the auxiliary Hessian
            # supplies each direction, the full quadrature supplies A*z.
            rhs=-residual;initial=step.copy();image=exact_action(initial);remaining=rhs-image;beta=np.linalg.norm(remaining);target=rtol*np.linalg.norm(rhs);outer=0
            if beta>target:
                vectors=[remaining/beta];directions=[];images=[];hessenberg=np.zeros((13,12))
                for k in range(12):
                    z,inner_info=gmres(operator,vectors[k],M=preconditioner,rtol=1e-4,atol=1e-14,restart=40,maxiter=15)
                    az=exact_action(z);w=az.copy()
                    for unused in range(2):
                        for i,v in enumerate(vectors):
                            value=float(v@w);hessenberg[i,k]+=value;w-=value*v
                    hessenberg[k+1,k]=np.linalg.norm(w);directions.append(z);images.append(az);right_hand=np.zeros(k+2);right_hand[0]=beta;coeff=np.linalg.lstsq(hessenberg[:k+2,:k+1],right_hand,rcond=None)[0]
                    step=initial+np.asarray(directions).T@coeff;remaining=rhs-image-np.asarray(images).T@coeff;outer=k+1
                    if np.linalg.norm(remaining)<=target:break
                    if hessenberg[k+1,k]<1e-14:break
                    vectors.append(w/hessenberg[k+1,k])
            row['full_Hessian_outer_iterations']=outer;row['full_Hessian_actions']=outer+1;row['full_linear_relative_residual']=float(np.linalg.norm(remaining)/np.linalg.norm(rhs))
        if not np.all(np.isfinite(step)):raise RuntimeError('nonfinite analytic orbital KKT step')
        norm=np.linalg.norm(step[:-1]);row['unscaled_step_norm']=float(norm)
        full_action=exact_action if exact_refinement and linearization_context is not None else action
        if residual_trust_region and norm>radius:
            step,trust_report=residual_subspace_step(residual,step,full_action,radius);row.update(trust_report);row['trust_radius']=radius;norm=np.linalg.norm(step[:-1])
        row['step_scale']=min(1.,radius/max(norm,1e-300))
        if checkpoint is not None:checkpoint(c,mu,row)
        step*=row['step_scale']
        accepted=False
        for fraction in [2.**(-k) for k in range(17)]:
            rotation=np.zeros((n,n));rotation[left,right]=fraction*step[:-1];rotation[right,left]=-fraction*step[:-1]
            trial=c@expm(rotation);newmu=mu+fraction*step[-1];ee,rr,_,_,ff,oo=evaluate(trial,newmu)
            following=abs(csf_overlap(c,trial)) if local_state_following else ff
            if following>=.7 and np.linalg.norm(rr)<np.linalg.norm(residual):
                c=trial;mu=newmu;accepted=True
                row['accepted_fraction']=fraction;row['accepted_residual_ratio']=float(np.linalg.norm(rr)/np.linalg.norm(residual))
                if residual_trust_region:
                    radius=max(1e-5,min(maximum_step,radius*(2. if fraction==1. else .5)))
                break
        if not accepted:raise RuntimeError(f'analytic ROKS KKT line search stalled: {row}')
    raise RuntimeError(f'analytic ROKS KKT residual did not converge: {history[-1]}')
