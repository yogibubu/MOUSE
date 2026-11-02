"""Residual trust-region and curvature certificates for orbital stationarity."""
import numpy as np
from scipy.linalg import svd,eigh
from scipy.sparse.linalg import LinearOperator,gmres,eigsh


def residual_trust_step(gradient,action,newton,radius,previous=None):
    """Minimize the full linearized residual in an orthonormal search subspace.

    Only numerical search directions are compressed, never the orbital space
    or the physical Hessian. The exact adjoint supplies residual descent even
    away from stationarity, where the moving-frame Jacobian is nonsymmetric.
    """
    basis=[]
    candidates=[newton,-action.adjoint(gradient)]
    if previous is not None:candidates.append(previous)
    for candidate in candidates:
        value=np.array(candidate,dtype=float,copy=True);norm=np.linalg.norm(value)
        if not np.isfinite(norm) or norm<1e-30:continue
        value/=norm
        for unused in range(2):
            for other in basis:value-=other*float(other@value)
        norm=np.linalg.norm(value)
        if norm>1e-10:basis.append(value/norm)
    if not basis:return np.zeros_like(gradient),0.,0
    q=np.column_stack(basis);jq=np.column_stack([action(v) for v in basis])
    u,s,vh=svd(jq,full_matrices=False);right=u.T@gradient
    def solve(shift):
        denominator=s*s+shift;keep=s>max(1e-30,1e-12*s[0]) if shift==0 else denominator>0
        return -vh.T@np.divide(s*right,denominator,out=np.zeros_like(s),where=keep)
    y=solve(0.)
    if np.linalg.norm(y)>radius:
        lo=0.;hi=max(float(s[0]**2),1e-12)
        while np.linalg.norm(solve(hi))>radius:hi*=4
        for unused in range(70):
            middle=.5*(lo+hi)
            if np.linalg.norm(solve(middle))>radius:lo=middle
            else:hi=middle
        y=solve(hi)
    step=q@y;predicted=float(gradient@gradient-np.linalg.norm(gradient+jq@y)**2)
    return step,predicted,len(basis)


def curvature_certificate(action,dimension,threshold=1e-6):
    """Count negative orbital modes from the complete stationary operator."""
    if not dimension:return dict(index=0,near_zero_modes=0,threshold_hartree=threshold,method='Empty orbital tangent')
    if dimension<=192:
        matrix=np.column_stack([action(v) for v in np.eye(dimension)])
        asymmetry=float(np.max(abs(matrix-matrix.T)))
        values=eigh(.5*(matrix+matrix.T),eigvals_only=True)
        return dict(index=int(np.sum(values< -threshold)),near_zero_modes=int(np.sum(abs(values)<=threshold)),
                    minimum_eigenvalue_hartree=float(values[0]),threshold_hartree=threshold,
                    Jacobian_asymmetry=asymmetry,method='Complete orbital tangent diagonalization')
    operator=LinearOperator((dimension,dimension),matvec=action.symmetric,dtype=float);k=min(8,dimension-1)
    while True:
        values,vectors=eigsh(operator,k=k,which='SA',tol=1e-9,maxiter=max(200,4*dimension),v0=np.random.default_rng(41).normal(size=dimension))
        order=np.argsort(values);values=values[order];vectors=vectors[:,order]
        if values[-1]>threshold or k==dimension-1:break
        k=min(dimension-1,2*k)
    residual=max(float(np.linalg.norm(operator@v-e*v)) for e,v in zip(values,vectors.T))
    if residual>1e-7:raise RuntimeError('stationary orbital curvature certificate not converged')
    if k==dimension-1 and values[-1]<=threshold:
        # Include the remaining mode explicitly; no unexamined negative mode.
        largest=eigsh(operator,k=1,which='LA',tol=1e-9,return_eigenvectors=False)[0]
        values=np.r_[values,largest]
    return dict(index=int(np.sum(values< -threshold)),near_zero_modes=int(np.sum(abs(values)<=threshold)),
                minimum_eigenvalue_hartree=float(values[0]),threshold_hartree=threshold,
                maximum_eigenpair_residual=residual,method='Complete negative prefix, matrix-free orbital operator')


def solve(surface,c,reference_overlap,*,tolerance,max_iterations,max_step,min_reference_overlap,
          min_step_overlap,step_overlap,callback,initial_radius=None,solver='trust'):
    radius=max_step if initial_radius is None else min(max_step,initial_radius)
    previous=None;data=None;records=[];status='maximum_iterations';linear_diagnostics={}
    for iteration in range(max_iterations+1):
        data=surface.evaluate(c) if data is None else surface.linearize(c,data)
        energy,g,action,*unused=data;residual=float(np.max(abs(g)));identity=reference_overlap(c)
        row=dict(iteration=iteration,energy_hartree=energy,orbital_residual=residual,reference_overlap=identity,
                 trust_radius=radius,**linear_diagnostics);records.append(row)
        if identity<min_reference_overlap:status='target_overlap_rejected';break
        callback(c,data,tuple(records),dict(status='converged' if residual<tolerance else 'iterating',trust_radius=radius,
                                            solver=solver,**surface.statistics))
        if residual<tolerance:status='converged';break
        if iteration==max_iterations:break
        if iteration>=12 and residual>.98*records[-11]['orbital_residual']:
            status='residual_stagnation';break
        operator=LinearOperator((len(g),len(g)),matvec=action,dtype=float)
        if solver=='trust':
            diagonal=action.preconditioner_diagonal
            scale=max(1e-4,.001*float(np.median(abs(diagonal))))
            denominator=np.where(abs(diagonal)<scale,np.where(diagonal<0,-scale,scale),diagonal)
            preconditioner=LinearOperator(operator.shape,matvec=lambda v:v/denominator,dtype=float)
            forcing=min(.1,max(1e-7,.02*np.sqrt(residual)))
        else:preconditioner=None;forcing=1e-5
        newton,info=gmres(operator,-g,M=preconditioner,rtol=forcing,atol=0.,restart=min(60,len(g)),maxiter=8)
        linear_diagnostics=dict(linear_solver_info=int(info),linear_relative_residual=float(np.linalg.norm(action(newton)+g)/np.linalg.norm(g)))
        accepted=False
        if solver=='newton':
            for direction in [newton,-action(g)]:
                norm=np.linalg.norm(direction)
                if not np.isfinite(norm) or norm<1e-16:continue
                direction=direction*min(1.,max_step/norm)
                for fraction in [1.,.5,.25,.125,.0625,.03125]:
                    step=fraction*direction;trial=surface.rotate(c,step)
                    if reference_overlap(trial)<min_reference_overlap or step_overlap(c,trial)<min_step_overlap:continue
                    trial_data=surface.evaluate(trial,linearize=False)
                    if np.linalg.norm(trial_data[1])<(1-1e-4*fraction)*np.linalg.norm(g):
                        c=trial;data=trial_data;previous=step;accepted=True;break
                if accepted:break
        else:
            for attempt in range(14):
                step,predicted,rank=residual_trust_step(g,action,newton,radius,previous)
                if predicted<=0 or np.linalg.norm(step)<1e-14:break
                trial=surface.rotate(c,step)
                allowed=reference_overlap(trial)>=min_reference_overlap and step_overlap(c,trial)>=min_step_overlap
                trial_data=surface.evaluate(trial,linearize=False) if allowed else None
                actual=float(g@g-np.linalg.norm(trial_data[1])**2) if allowed else -np.inf
                ratio=actual/predicted
                if ratio<.25:radius*=.5
                elif ratio>.75 and np.linalg.norm(step)>.8*radius:radius=min(max_step,2*radius)
                if allowed and ratio>.05 and actual>0:
                    c=trial;data=trial_data;previous=step;accepted=True
                    linear_diagnostics.update(trust_ratio=float(ratio),trust_subspace_rank=rank,trust_attempts=attempt+1)
                    break
        if not accepted:status='trust_region_exhausted' if solver=='trust' else 'line_search_exhausted';break
        if radius<1e-10*max_step and float(np.max(abs(data[1])))>tolerance:
            # Accepting changes below numerical resolution cannot certify a root.
            c_data=surface.linearize(c,data);records.append(dict(iteration=iteration+1,energy_hartree=c_data[0],orbital_residual=float(np.max(abs(c_data[1]))),reference_overlap=reference_overlap(c),trust_radius=radius))
            data=c_data;status='residual_stagnation';break
    return c,data,tuple(records),dict(status=status,solver=solver,trust_radius=radius,**surface.statistics)


def relax_closed_orbitals(surface,c,*,max_step,callback,max_iterations=30,tolerance=1e-6):
    """Initialize the same energy by relaxing closed/virtual rotations only.

    The two open spatial orbitals stay fixed during this initialization. The
    final stationarity search and nuclear response always release all rotations.
    """
    occ=surface.occupations;total=occ.sum(axis=0);left,right=surface.pairs[0]
    selected=np.flatnonzero(((total[left]==2)&(total[right]==0))|((total[left]==0)&(total[right]==2)))
    rows=[];active=np.flatnonzero(total==1);original_active=c[:,active].copy()
    for iteration in range(max_iterations+1):
        data=surface.evaluate(c);energy,g,action,*unused=data
        rows.append(dict(iteration=iteration,phase='closed_orbital_initialization',energy_hartree=energy,
                         orbital_residual=float(np.max(abs(g)))))
        callback(c,data,tuple(rows),dict(status='initializing',solver='trust',trust_radius=max_step,**surface.statistics))
        if not len(selected) or np.max(abs(g[selected]))<tolerance or iteration==max_iterations:break
        def embed(v):
            full=np.zeros(surface.dimension);full[selected]=v;return full
        operator=LinearOperator((len(selected),len(selected)),matvec=lambda v:action(embed(v))[selected],dtype=float)
        diagonal=action.preconditioner_diagonal[selected]
        preconditioner=LinearOperator(operator.shape,matvec=lambda v:v/np.maximum(abs(diagonal),1e-4),dtype=float)
        step,info=gmres(operator,-g[selected],M=preconditioner,rtol=1e-3,atol=0.,restart=min(60,len(selected)),maxiter=5)
        if not np.all(np.isfinite(step)) or g[selected]@step>=0:step=-g[selected]/np.maximum(abs(diagonal),1e-4)
        step=embed(step);step*=min(1.,max_step/max(np.linalg.norm(step),1e-30));accepted=False
        for fraction in [1.,.5,.25,.125,.0625,.03125,.015625,.0078125]:
            trial=surface.rotate(c,fraction*step);trial_data=surface.evaluate(trial,linearize=False)
            if trial_data[0]<energy+1e-4*fraction*(g@step):c=trial;accepted=True;break
        if not accepted:break
    if np.max(abs(c[:,active]-original_active))>1e-10:raise RuntimeError('open orbitals moved during frozen initialization')
    return c,tuple(rows)
