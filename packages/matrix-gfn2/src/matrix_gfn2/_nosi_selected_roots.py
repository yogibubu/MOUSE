"""Exact selected generalized CI roots with a sparse metric and rank-one projection."""
import numpy as np
from scipy.sparse import csc_matrix,eye
from scipy.sparse.linalg import splu,LinearOperator,eigsh,lobpcg
from scipy.linalg import eigh


def metric_certificate(metric,nbase,shift=1e-8):
    """Certify ground-projected S - shift I by its canonical Schur complement."""
    n=len(metric);p=metric[0,1:]/metric[0,0]
    projected=metric[1:,1:]-np.outer(metric[1:,0],p)
    bound=float(np.min(2*np.diag(projected)-np.sum(abs(projected),axis=1)))
    if bound>shift:return dict(method='Gershgorin',lower_bound=bound,certified_shift=shift)
    g=csc_matrix(metric[nbase:,nbase:])-shift*eye(n-nbase,format='csc')
    gb=float(np.min(g.diagonal()-np.asarray(abs(g).sum(axis=1)).ravel()+abs(g.diagonal())))
    if gb>0:
        border=metric[:nbase,:nbase].copy();border.flat[::nbase+1]-=np.r_[0.,np.full(nbase-1,shift)]
        cross=metric[nbase:,:nbase];small=border-cross.T@splu(g).solve(cross);minimum=float(eigh(small,eigvals_only=True)[0])
        if minimum>0:return dict(method='Canonical Gershgorin plus exact border Schur complement',lower_bound=None,certified_shift=shift,canonical_shifted_lower_bound=gb,border_Schur_minimum=minimum)
    smallest=float(eigh(projected,subset_by_index=[0,0],eigvals_only=True)[0])
    if smallest<=shift:raise ValueError(('unresolved CI metric dependence',smallest))
    return dict(method='Smallest projected eigenvalue',lower_bound=smallest,certified_shift=shift)


def solve(h,metric,ground_energy,roots):
    """ARPACK solves the full projected pencil; only the returned roots are selected.

    The canonical metric block is sparse. Ground removal is a rank-one
    update, inverted by the exact Sherman--Morrison identity. No metric
    entries, Hamiltonian elements or configurations are discarded.
    """
    n=len(h)-1;k=min(roots,n);p=metric[0,1:]/metric[0,0]
    sp=metric[1:,1:]-np.outer(metric[1:,0],p)
    hp=h[1:,1:]-np.outer(h[1:,0],p)-np.outer(p,h[0,1:])+h[0,0]*np.outer(p,p)
    if k>=n-1:
        energies,z=eigh(hp,sp,subset_by_index=[0,k-1]);method='Dense selected generalized eigensolve'
    else:
        sparse=csc_matrix(metric[1:,1:]);factor=splu(sparse);u=metric[1:,0]/np.sqrt(metric[0,0]);v=factor.solve(u);denominator=1-float(u@v)
        if denominator<=0:raise ValueError('indefinite ground-projected metric')
        def metric_action(x):return sparse@x-u*(u@x)
        def inverse_action(x):
            y=factor.solve(x);return y+v*((u@y)/denominator)
        m=LinearOperator((n,n),matvec=metric_action,dtype=float);mi=LinearOperator((n,n),matvec=inverse_action,dtype=float)
        shifted=hp-ground_energy*sp
        values,z=eigsh(shifted,k=k,M=m,Minv=mi,which='SA',tol=1e-11,ncv=min(n,max(2*k+1,100)),v0=np.random.default_rng(715).normal(size=n),maxiter=10000)
        order=np.argsort(values);z=z[:,order];energies=values[order]+ground_energy;method='ARPACK with exact sparse metric inverse and rank-one ground projection'
    residual=hp@z-(sp@z)*energies;maximum=float(np.max(np.linalg.norm(residual,axis=0)));orthogonal=float(np.max(abs(z.T@sp@z-np.eye(k))))
    if maximum>1e-8 or orthogonal>1e-8:raise ValueError(('CI root residual qualification failed',maximum,orthogonal))
    ci=np.vstack((-p@z,z))
    return energies,ci,dict(method=method,roots=k,maximum_residual_hartree=maximum,maximum_metric_orthonormality_error=orthogonal,configurations_discarded=0)


def projector_population(metric,indices,ci):
    """Exact metric projector onto a ground-projected subset of CI functions."""
    subset=csc_matrix(metric[np.ix_(indices,indices)]);factor=splu(subset)
    u=metric[indices,0]/np.sqrt(metric[0,0]);v=factor.solve(u);denominator=1-float(u@v)
    if denominator<=0:raise ValueError('dependent ground-projected diagnostic subspace')
    cross=(metric[indices]-np.outer(metric[indices,0]/metric[0,0],metric[0]))@ci
    solved=factor.solve(cross);solved+=np.outer(v,(u@solved)/denominator)
    return np.sum(cross*solved,axis=0)

def solve_full(h,metric,roots,*,energy_shift=None,algorithm='arpack'):
    """Solve the same full CI pencil, including its interacting ground state."""
    n=len(h);k=min(int(roots),n)
    if k<1:raise ValueError('positive number of full CI roots required')
    if algorithm not in ('arpack','lobpcg'):raise ValueError('unknown full CI eigensolver')
    shift=float(h[0,0]/metric[0,0]) if energy_shift is None else float(energy_shift)
    if k>=n-1 or (algorithm=='lobpcg' and n<5*k):
        energies,z=eigh(h,metric,subset_by_index=[0,k-1]);method='Dense full generalized eigensolve'
    elif algorithm=='lobpcg':
        sparse=csc_matrix(metric)
        action=LinearOperator((n,n),matvec=lambda x:h@x-shift*(sparse@x),matmat=lambda x:h@x-shift*(sparse@x),dtype=float)
        diagonal=np.maximum(abs(np.diag(h)-shift*np.diag(metric)),1e-6)
        preconditioner=LinearOperator((n,n),matvec=lambda x:x/diagonal,matmat=lambda x:x/diagonal[:,None],dtype=float)
        values,z=lobpcg(action,np.random.default_rng(715).normal(size=(n,k)),B=sparse,M=preconditioner,largest=False,tol=1e-10,maxiter=500)
        order=np.argsort(values);z=z[:,order];energies=values[order]+shift;method='Block LOBPCG with full generalized metric and diagonal preconditioner'
    else:
        sparse=csc_matrix(metric);factor=splu(sparse)
        m=LinearOperator((n,n),matvec=lambda x:sparse@x,dtype=float)
        mi=LinearOperator((n,n),matvec=factor.solve,dtype=float)
        action=LinearOperator((n,n),matvec=lambda x:h@x-shift*(sparse@x),dtype=float)
        values,z=eigsh(action,k=k,M=m,Minv=mi,which='SA',tol=1e-11,ncv=min(n,max(2*k+1,100)),v0=np.random.default_rng(715).normal(size=n),maxiter=10000)
        order=np.argsort(values);z=z[:,order];energies=values[order]+shift;method='ARPACK full CI with exact sparse metric inverse'
    residual=h@z-(metric@z)*energies;error=float(np.max(np.linalg.norm(residual,axis=0)));orthogonal=float(np.max(abs(z.T@metric@z-np.eye(k))))
    if max(error,orthogonal)>1e-8:raise ValueError(('full CI residual qualification failed',error,orthogonal))
    return energies,z,dict(method=method,roots=k,maximum_residual_hartree=error,maximum_metric_orthonormality_error=orthogonal,configurations_discarded=0)
