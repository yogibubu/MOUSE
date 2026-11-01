"""Analytic Newton--Krylov stationarity of one ground-orthogonal ROKS CSF.

The residual and its action are the complete orbital KKT equations. Neither
orbital rotations nor density response are truncated; finite differences do
not enter the solver. Full dense orbital Jacobians are never allocated.
"""
from types import SimpleNamespace
import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator, gmres, minres
from .roks_orbital_response import orbital_residual_action
from ._nosi_orthogonality import ground_overlap_jet, ground_overlap_action
from ._nosi_refine import normalize


def solve(context, coefficients, ground, transition, multiplier=0., *,
          tolerance=1e-10, maximum_iterations=80, maximum_step=.2, checkpoint=None, linearization_context=None,
          local_state_following=False, krylov='gmres'):
    c=normalize(coefficients,context.state.overlap);n=len(c);no=context.state.electron_count//2
    occ=np.zeros((2,n));occ[:,:no]=1;occ[0,0]=0;occ[0,no]=1
    triplet=np.array([np.maximum(*occ),np.minimum(*occ)])
    left,right=np.triu_indices(n,1);labels=np.vstack([occ,triplet]);keep=np.any(labels[:,left]!=labels[:,right],axis=0);left,right=left[keep],right[keep]
    linear=context if linearization_context is None else linearization_context
    mu=float(multiplier);history=[];s=context.state.overlap
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
        error=float(np.max(abs(residual)));row=dict(iteration=iteration,energy_hartree=energy,residual=error,ground_overlap=overlap,fidelity=fidelity,multiplier=mu);history.append(row)
        if checkpoint is not None:checkpoint(c,mu,row)
        if error<tolerance:return c,mu,dict(converged=True,history=history,energy_hartree=energy,residual=error,ground_overlap=overlap,fidelity=fidelity,retained_transition_rank=reference.shape[1],state_following='accepted-step coherent CSF overlap' if local_state_following else 'anchor transition subspace',krylov=krylov,complete_orbital_rotations=len(left),dense_orbital_Jacobian_allocated=False,full_functional_residual=True,auxiliary_linearization=linearization_context is not None)
        if iteration==maximum_iterations:break
        state=SimpleNamespace(**vars(context.state));state.coefficients=c;state.alpha_occupations,state.beta_occupations=occ
        kernel=linear.density_response_kernel(state)
        def action(vector):
            x=vector[:-1];result=np.zeros(len(left))
            for weight,o,f in components:result+=weight*orbital_residual_action(c,o,f,kernel,linear.polarization,left,right,x,symmetric=True)
            if mu:result-=mu*ground_overlap_action(ground,c,s,occ,left,right,x)[1]
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
        elif krylov=='minres':step,info=minres(operator,-residual,M=preconditioner,rtol=rtol,maxiter=600,callback=counted)
        else:raise ValueError('unsupported orbital Krylov solver')
        row['Krylov_iterations']=linear_iterations[0];row['Krylov_info']=int(info)
        if not np.all(np.isfinite(step)):raise RuntimeError('nonfinite analytic orbital KKT step')
        norm=np.linalg.norm(step[:-1]);step*=min(1.,maximum_step/max(norm,1e-300))
        accepted=False
        for fraction in [1.,.5,.25,.125,.0625,.03125,.015625]:
            rotation=np.zeros((n,n));rotation[left,right]=fraction*step[:-1];rotation[right,left]=-fraction*step[:-1]
            trial=c@expm(rotation);newmu=mu+fraction*step[-1];ee,rr,_,_,ff,oo=evaluate(trial,newmu)
            following=abs(csf_overlap(c,trial)) if local_state_following else ff
            if following>=.7 and np.linalg.norm(rr)<np.linalg.norm(residual):
                c=trial;mu=newmu;accepted=True;break
        if not accepted:raise RuntimeError(f'analytic ROKS KKT line search stalled: {row}')
    raise RuntimeError(f'analytic ROKS KKT residual did not converge: {history[-1]}')
