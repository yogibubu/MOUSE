"""Positive static response constrained by actual particle--hole amplitudes."""
import numpy as np

def canonical_frontier_indices(transitions,no,nvirtual,frontier,*,redundancy_threshold=.7):
    """Canonical singlets retained by the CID overlap rule, without a CI run."""
    t=np.asarray(transitions,dtype=float)
    if not 1<=frontier<=min(no,nvirtual) or t.ndim!=2 or t.shape[0]!=no*nvirtual:
        raise ValueError('transition chart incompatible with frontier')
    indices=np.array([h*nvirtual+a for h in range(no-frontier,no) for a in range(frontier)],dtype=int)
    overlap=np.max(abs(t[indices]),axis=1,initial=0.)
    return indices[overlap<redundancy_threshold]

def constrained_vertices(vertices,gaps,canonical_indices,transitions,*,tolerance=1e-10):
    """Apply the exact gap-metric constraint, without a hole/particle product.

    ``vertices`` is 2 L D^-1/2. Canonical coordinates and coherent columns
    in ``transitions`` span the excluded response directions. The result
    is B (I-U U.T), where U orthonormalizes D^-1/2 times that span.
    """
    gap=np.asarray(gaps,dtype=float).ravel();b=np.asarray(vertices,dtype=float)
    if b.ndim!=2 or b.shape[1]!=len(gap) or np.any(gap<=0):raise ValueError('incompatible positive transition gaps')
    indices=np.unique(np.asarray(canonical_indices,dtype=int))
    if np.any(indices<0) or np.any(indices>=len(gap)):raise ValueError('active transition index outside chart')
    t=np.asarray(transitions,dtype=float)
    if t.ndim!=2 or t.shape[0]!=len(gap):raise ValueError('coherent transition columns have incompatible dimensions')
    c=t/np.sqrt(gap)[:,None];c[indices]=0.
    u,s,vh=np.linalg.svd(c,full_matrices=False);keep=s>tolerance
    u=u[:,keep];result=b.copy();result[:,indices]=0.
    if len(u.T):result-=(result@u)@u.T
    error=float(np.linalg.norm(result@u)/max(np.linalg.norm(result),1e-300)) if len(u.T) else 0.
    if error>1e-9:raise ValueError('unresolved active response constraint')
    return result,dict(canonical_directions=len(indices),coherent_extra_directions=int(keep.sum()),total_excluded_directions=int(len(indices)+keep.sum()),constraint_relative_residual=error,coherent_singular_values=s.tolist(),positive_polarization_by_Gram_construction=True)
