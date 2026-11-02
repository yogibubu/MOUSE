"""Rank-revealing generalized Davidson for complete prepared CI pencils."""
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csc_matrix
from concurrent.futures import ThreadPoolExecutor
try:
    from threadpoolctl import threadpool_limits
except ImportError:
    from contextlib import nullcontext
    def threadpool_limits(unused):return nullcontext()


def lowest(h,metric,roots,shift,initial=None,*,tolerance=1e-10,maxiter=300,block_workers=1):
    n=len(h);k=int(roots);s=csc_matrix(metric);diagonal=np.diag(h)-shift*np.diag(metric);sdiag=np.diag(metric)
    if block_workers<1:raise ValueError('positive row-block worker count required')
    def action(x):
        if block_workers==1 or n<256:return h@x-shift*(s@x)
        result=np.empty((n,x.shape[1]));edges=np.linspace(0,n,block_workers+1,dtype=int)
        def rows(pair):
            lo,hi=pair;np.matmul(h[lo:hi],x,out=result[lo:hi])
        # Share the immutable Hamiltonian; do not duplicate its large array.
        with threadpool_limits(1),ThreadPoolExecutor(max_workers=block_workers) as pool:
            list(pool.map(rows,zip(edges[:-1],edges[1:])))
        return result-shift*(s@x)
    def orthogonalize(x,v=None):
        x=x.copy()
        if v is not None:
            for unused in range(2):x-=v@(v.T@(s@x))
        gram=x.T@(s@x);values,u=eigh((gram+gram.T)/2)
        keep=values>max(1e-26,float(values[-1])*1e-12)
        return x@(u[:,keep]/np.sqrt(values[keep])[None,:])
    if initial is None:
        ids=np.argsort(diagonal/np.diag(metric))[:min(n,max(k+8,2*k))];guess=np.zeros((n,len(ids)));guess[ids,np.arange(len(ids))]=1.
    else:
        guess=np.asarray(initial,dtype=float)
        if guess.ndim!=2 or guess.shape[0]!=n or guess.shape[1]<k:raise ValueError('initial Davidson space has incompatible dimensions')
    v=orthogonalize(guess);av=action(v);limit=min(n,max(160,6*k+40));restarts=0
    for iteration in range(maxiter):
        projected=v.T@av;energies,u=eigh((projected+projected.T)/2)
        vectors=v@u[:,:k];sv=s@vectors;residual=av@u[:,:k]-sv*energies[:k]
        norms=np.linalg.norm(residual,axis=0)
        if np.max(norms)<tolerance:
            return energies[:k]+shift,vectors,dict(iterations=iteration+1,restarts=restarts,maximum_residual_hartree=float(np.max(norms)),largest_search_space=limit)
        active=np.flatnonzero(norms>tolerance)
        denominator=diagonal[:,None]-sdiag[:,None]*energies[None,active]
        denominator=np.where(abs(denominator)<1e-5,np.where(denominator<0,-1e-5,1e-5),denominator)
        correction=-residual[:,active]/denominator
        if len(v.T)+len(correction.T)>limit:
            keep=min(len(u.T),k+12);v=v@u[:,:keep];av=av@u[:,:keep];restarts+=1
        correction=orthogonalize(correction,v)
        if not len(correction.T):raise ValueError('Davidson search stalled before full residual convergence')
        v=np.column_stack((v,correction));av=np.column_stack((av,action(correction)))
    raise ValueError(('Davidson did not converge',float(np.max(norms))))
