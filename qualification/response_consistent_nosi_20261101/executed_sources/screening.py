"""Parameter-free static constrained screening in a prepared native basis."""
import numpy as np
from scipy.linalg import solve

def sector(vectors,dimension):
    columns=[]
    for v in vectors:
        x=np.array(v,copy=True)
        for _ in range(2):
            for q in columns:x-=q*(q@x)
        n=np.linalg.norm(x)
        if n>1e-8:columns.append(x/n)
    return np.column_stack(columns) if columns else np.empty((dimension,0))

def screened(eri,canonical,levels,no,holes=(),particles=()):
    n=len(canonical);nv=n-no
    oh=sector([np.eye(no)[:,j] for j in [no-2,no-1]]+list(holes),no)
    op=sector([np.eye(nv)[:,j] for j in [0,1]]+list(particles),nv)
    wh=np.sum(oh**2,axis=1);wp=np.sum(op**2,axis=1)
    excluded=1-wh[:,None]*wp[None,:];excluded=np.clip(excluded,0,1)
    ii=np.repeat(np.arange(no),nv);aa=np.tile(np.arange(no,n),no);gap=levels[aa]-levels[ii];assert np.min(gap)>0
    keep=excluded.ravel()>1e-10;weights=excluded.ravel()[keep];ii=ii[keep];aa=aa[keep];gap=gap[keep]
    transition=np.einsum('pi,qi->ipq',canonical[:,ii],canonical[:,aa]);transition=transition+transition.transpose(0,2,1)
    t=transition.reshape(len(transition),-1).T*np.sqrt(weights)[None,:]
    v=eri.reshape(n*n,n*n);vt=v@t;denominator=np.diag(gap)+t.T@vt
    w=v-vt@solve(denominator,vt.T,assume_a='pos');w=.5*(w+w.T)
    result=w.reshape(eri.shape)
    assert max(np.max(abs(result-result.swapaxes(0,1))),np.max(abs(result-result.swapaxes(2,3))))<1e-8
    return result,dict(polarization_terms=len(gap),excluded_weights=excluded,active_hole_dimension=oh.shape[1],active_particle_dimension=op.shape[1],minimum_gap=float(np.min(gap)))
