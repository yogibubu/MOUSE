"""Zero-frequency subtraction for an extended, nonorthogonal response pencil.

This is an experimental model, not the default OO-CID Hamiltonian.  The
Hamiltonian must contain excitation energies relative to the native ground.
The P-space static response is preserved when a complementary Q space is
coupled to it.  No constant excitation-energy offset is introduced.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator, cg, eigsh


def static_subtraction(h, metric, primary, *, tolerance=1e-10):
    """Return R, Delta with H_sub = H + R Delta R.T.

    Q is first made metric-orthogonal to P.  In those coordinates the
    correction is V D^-1 V.T in the P block.  D is the complete interacting
    Q Hamiltonian, not its diagonal.  Its inverse is applied iteratively;
    no configurations or matrix entries are dropped.
    """
    h=np.asarray(h,dtype=float);s=np.asarray(metric,dtype=float)
    if h.ndim!=2 or h.shape[0]!=h.shape[1] or s.shape!=h.shape:
        raise ValueError('matching square Hamiltonian and metric required')
    p=np.asarray(primary,dtype=int)
    if p.ndim!=1 or not len(p) or len(np.unique(p))!=len(p) or np.any(p<0) or np.any(p>=len(h)):
        raise ValueError('distinct primary indices inside the pencil required')
    q=np.setdiff1d(np.arange(len(h)),p)
    if not len(q):raise ValueError('nonempty complementary space required')
    spp=s[np.ix_(p,p)];factor=cho_factor(spp)
    x=cho_solve(factor,s[np.ix_(p,q)])
    a=h[np.ix_(p,p)];b=h[np.ix_(p,q)];d0=csc_matrix(h[np.ix_(q,q)])
    v=b-a@x
    def action(z):
        xz=x@z
        return d0@z-b.T@xz-x.T@(b@z)+x.T@(a@xz)
    d=LinearOperator((len(q),len(q)),matvec=action,dtype=float)
    if len(q)<3:
        smallest=float(np.linalg.eigvalsh(action(np.eye(len(q))))[0])
    else:
        smallest=float(eigsh(d,k=1,which='SA',return_eigenvectors=False,tol=1e-9,v0=np.random.default_rng(931).normal(size=len(q)))[0])
    if smallest<=0:raise ValueError(('static subtraction requires positive complementary excitation Hamiltonian',smallest))
    diagonal=d0.diagonal()-2*np.sum(b*x,axis=0)+np.sum(x*(a@x),axis=0)
    if np.any(diagonal<=0):raise ValueError('nonpositive complementary preconditioner')
    preconditioner=LinearOperator(d.shape,matvec=lambda z:z/diagonal,dtype=float)
    solution=np.empty((len(q),len(p)));residuals=[];iterations=[]
    for i,rhs in enumerate(v):
        count=[0]
        def callback(z):count[0]+=1
        solution[:,i],info=cg(d,rhs,M=preconditioner,rtol=tolerance,atol=tolerance*1e-3,maxiter=3000,callback=callback)
        error=float(np.linalg.norm(action(solution[:,i])-rhs)/max(np.linalg.norm(rhs),1e-15))
        if info or error>10*tolerance:raise ValueError(('complementary static inverse did not converge',i,info,error))
        residuals.append(error);iterations.append(count[0])
    delta=v@solution;asymmetry=float(np.max(abs(delta-delta.T)));delta=.5*(delta+delta.T)
    if asymmetry>100*tolerance:raise ValueError(('asymmetric static counterterm',asymmetry))
    r=np.empty((len(h),len(p)));r[p]=np.eye(len(p));r[q]=x.T
    return r,delta,dict(definition='Exact zero-frequency self-energy subtraction',primary_dimension=len(p),complement_dimension=len(q),complement_minimum_eigenvalue_hartree=smallest,maximum_inverse_relative_residual=max(residuals),inverse_iterations=iterations,counterterm_asymmetry_hartree=asymmetry,energy_offset_ev=0.,diagonal_complement_approximation=False)
