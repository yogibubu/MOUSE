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


def residual_subspace_step(residual,newton,action,radius,*,directions=(),gradient_action=None):
    """Minimize the linearized residual in an augmented Newton trust subspace.

    The covariant KKT action gives the gradient of the squared residual
    norm. The moving-frame form has the same action on the residual
    because its skew connection annihilates that vector. Only the orbital part is radius constrained; the scalar
    Lagrange multiplier is optimized jointly. Optional Krylov directions
    enlarge the Newton/Cauchy basis without forming a dense Jacobian.
    No stationary equation or energy parameter is modified.
    """
    gradient=(action if gradient_action is None else gradient_action)(residual)
    candidates=[newton,-gradient,*directions];basis=[]
    for vector in candidates:
        value=np.asarray(vector,dtype=float).copy();norm=np.linalg.norm(value)
        if norm<1e-30:continue
        value/=norm
        for unused in range(2):
            for previous in basis:value-=previous*float(previous@value)
        norm=np.linalg.norm(value)
        if norm>1e-10:basis.append(value/norm)
    q=np.column_stack(basis) if basis else np.empty((len(residual),0))
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


def retract_ground_overlap(coefficients,ground,overlap,occupations,left,right,*,tolerance=1e-12):
    """Restore the original singlet/ground constraint with orbital rotations.

    This is a geometric retraction, not a projected-energy replacement.
    It requires determinant cofactors only, without Fock or XC evaluations.
    """
    c=np.asarray(coefficients).copy();n=len(c)
    for iteration in range(24):
        value,j=ground_overlap_jet(ground,c,overlap,occupations,left,right)
        if abs(value)<tolerance:return c
        norm2=float(j@j)
        if norm2<1e-26:raise RuntimeError('singular ground-overlap retraction')
        step=-value*j/norm2;norm=np.linalg.norm(step)
        step*=min(1.,.1/max(norm,1e-300))
        for fraction in [2.**(-k) for k in range(14)]:
            rotation=np.zeros((n,n));rotation[left,right]=fraction*step;rotation[right,left]=-fraction*step
            trial=c@expm(rotation)
            candidate=ground_overlap_jet(ground,trial,overlap,occupations,left,right)[0]
            if abs(candidate)<abs(value):c=trial;break
        else:raise RuntimeError('ground-overlap retraction line search stalled')
    raise RuntimeError('ground-overlap retraction did not converge')


def solve(context, coefficients, ground, transition, multiplier=0., *,
          tolerance=1e-10, maximum_iterations=80, maximum_step=.2, checkpoint=None, linearization_context=None,
          local_state_following=False, krylov='gmres',exact_refinement=False,residual_trust_region=False,
          trust_krylov_dimension=0,retract_constraint=False,moving_frame_jacobian=False,initial_trust_radius=None):
    if moving_frame_jacobian and krylov!='gmres':raise ValueError('moving-frame Jacobian requires GMRES')
    if not 0<=trust_krylov_dimension<=64:raise ValueError('invalid trust Krylov dimension')
    c=normalize(coefficients,context.state.overlap);n=len(c);no=context.state.electron_count//2
    occ=np.zeros((2,n));occ[:,:no]=1;occ[0,0]=0;occ[0,no]=1
    triplet=np.array([np.maximum(*occ),np.minimum(*occ)])
    left,right=np.triu_indices(n,1);labels=np.vstack([occ,triplet]);keep=np.any(labels[:,left]!=labels[:,right],axis=0);left,right=left[keep],right[keep]
    linear=context if linearization_context is None else linearization_context
    mu=float(multiplier);history=[];s=context.state.overlap;radius=maximum_step if initial_trust_radius is None else min(maximum_step,float(initial_trust_radius));accepted_diagnostics={}
    if not np.isfinite(radius) or radius<=0:raise ValueError('invalid initial trust radius')
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
    if retract_constraint:c=retract_ground_overlap(c,ground,s,occ,left,right)
    for iteration in range(maximum_iterations+1):
        energy,residual,components,j,fidelity,overlap=evaluate(c,mu)
        if retract_constraint and j@j>1e-26:
            adjustment=float(j@residual[:-1]/(j@j));mu+=adjustment;residual[:-1]-=adjustment*j
        error=float(np.max(abs(residual)));row=dict(iteration=iteration,trust_radius=float(radius),energy_hartree=energy,residual=error,residual_l2=float(np.linalg.norm(residual)),ground_overlap=overlap,fidelity=fidelity,multiplier=mu);row.update(accepted_diagnostics);history.append(row)
        if checkpoint is not None:checkpoint(c,mu,row)
        if error<tolerance:return c,mu,dict(converged=True,history=history,energy_hartree=energy,residual=error,ground_overlap=overlap,fidelity=fidelity,retained_transition_rank=reference.shape[1],state_following='accepted-step coherent CSF overlap' if local_state_following else 'anchor transition subspace',krylov=krylov,complete_orbital_rotations=len(left),dense_orbital_Jacobian_allocated=False,full_functional_residual=True,auxiliary_linearization=linearization_context is not None,trust_krylov_dimension=trust_krylov_dimension,ground_constraint_retracted=retract_constraint,moving_frame_Jacobian=moving_frame_jacobian)
        if iteration==maximum_iterations:break
        state=SimpleNamespace(**vars(context.state));state.coefficients=c;state.alpha_occupations,state.beta_occupations=occ
        kernel=linear.density_response_kernel(state)
        constraint=prepare_ground_overlap_action(ground,c,s,occ,left,right) if mu else None
        search_directions=[]
        residual_matrix=np.zeros((n,n));residual_matrix[left,right]=residual[:-1];residual_matrix[right,left]=-residual[:-1]
        def connection(vector):
            rotation=np.zeros((n,n));rotation[left,right]=vector[:-1];rotation[right,left]=-vector[:-1]
            return np.r_[.5*(rotation@residual_matrix-residual_matrix@rotation)[left,right],0.]
        def action(vector):
            if trust_krylov_dimension and len(search_directions)<max(2*trust_krylov_dimension,40):
                norm=np.linalg.norm(vector)
                if norm>1e-20:search_directions.append(np.asarray(vector).copy()/norm)
            x=vector[:-1];result=roks_residual_action(c,components,kernel,linear.polarization,left,right,x)
            if constraint is not None:result-=mu*constraint(x)[1]
            return np.r_[result-vector[-1]*j,-j@x]-(connection(vector) if moving_frame_jacobian else 0.)
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
        if checkpoint is not None:checkpoint(c,mu,row)
        if exact_refinement and linearization_context is not None:
            exact_kernel=context.density_response_kernel(state)
            def exact_action(vector):
                x=vector[:-1];result=roks_residual_action(c,components,exact_kernel,context.polarization,left,right,x)
                if constraint is not None:result-=mu*constraint(x)[1]
                return np.r_[result-vector[-1]*j,-j@x]-(connection(vector) if moving_frame_jacobian else 0.)
            if residual_trust_region and np.linalg.norm(step[:-1])>radius:
                row['full_refinement_skipped_for_trust_step']=True
            else:
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
            proposal_action=action if trust_krylov_dimension and linearization_context is not None else full_action
            step,trust_report=residual_subspace_step(residual,step,proposal_action,radius,directions=search_directions[:trust_krylov_dimension],gradient_action=full_action);row.update(trust_report);row['trust_radius']=radius;row['trust_proposal_uses_auxiliary_response']=proposal_action is action and linearization_context is not None;norm=np.linalg.norm(step[:-1])
        row['step_scale']=min(1.,radius/max(norm,1e-300))
        step*=row['step_scale']
        image=full_action(step) if residual_trust_region else None
        if image is not None:row['full_predicted_relative_residual']=float(np.linalg.norm(residual+image)/np.linalg.norm(residual))
        if checkpoint is not None:checkpoint(c,mu,row)
        accepted=False
        for fraction in [2.**(-k) for k in range(17)]:
            rotation=np.zeros((n,n));rotation[left,right]=fraction*step[:-1];rotation[right,left]=-fraction*step[:-1]
            trial=c@expm(rotation);newmu=mu+fraction*step[-1]
            if retract_constraint:trial=retract_ground_overlap(trial,ground,s,occ,left,right)
            ee,rr,_,trial_j,ff,oo=evaluate(trial,newmu)
            if retract_constraint and trial_j@trial_j>1e-26:
                adjustment=float(trial_j@rr[:-1]/(trial_j@trial_j));newmu+=adjustment;rr[:-1]-=adjustment*trial_j
            following=abs(csf_overlap(c,trial)) if local_state_following else ff
            if following>=.7 and np.linalg.norm(rr)<np.linalg.norm(residual):
                c=trial;mu=newmu;accepted=True
                row['accepted_fraction']=fraction;row['accepted_residual_ratio']=float(np.linalg.norm(rr)/np.linalg.norm(residual))
                if residual_trust_region:
                    predicted=float(np.dot(residual,residual)-np.dot(residual+fraction*image,residual+fraction*image))
                    actual=float(np.dot(residual,residual)-np.dot(rr,rr))
                    ratio=actual/predicted if predicted>0 else 0.
                    if ratio<.25 or fraction<.5:radius=max(1e-5,min(.25*radius,2*fraction*norm))
                    elif ratio>.75 and fraction*norm>=.8*radius:radius=min(maximum_step,2*radius)
                    accepted_diagnostics=dict(previous_step_model_ratio=ratio,previous_accepted_fraction=fraction,previous_actual_residual_reduction=actual,previous_predicted_residual_reduction=predicted)

                break
        if not accepted:raise RuntimeError(f'analytic ROKS KKT line search stalled: {row}')
    raise RuntimeError(f'analytic ROKS KKT residual did not converge: {history[-1]}')
