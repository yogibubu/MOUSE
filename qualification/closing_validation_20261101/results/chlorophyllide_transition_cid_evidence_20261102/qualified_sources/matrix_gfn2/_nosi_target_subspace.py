"""Track a fixed orthonormal target space through fragmented CI eigenstates."""
import numpy as np


def select_target_subspace(amplitudes,minimum_coverage=.49):
    """Retain roots until every target direction has the required coverage.

    Rows are overlaps of orthonormal target functions with orthonormal CI
    eigenstates. Coverage is the smallest eigenvalue of the target-space
    projector, so equivalent unitary target rotations give the same result.
    A target split over several eigenstates is not forced into one root.
    """
    a=np.asarray(amplitudes)
    if a.ndim!=2 or not a.shape[0] or a.shape[1]<a.shape[0] or not np.all(np.isfinite(a)):
        raise ValueError('invalid target/eigenstate overlap array')
    if not 0<minimum_coverage<=1:raise ValueError('invalid target coverage')
    complete=a@a.conj().T;full=np.linalg.eigvalsh(complete)
    if full[-1]>1+1e-7:raise ValueError('target and CI functions must be orthonormal')
    weights=np.sum(abs(a)**2,axis=0);selected=[];coverage=0.
    for root in np.argsort(-weights,kind='stable'):
        selected.append(int(root))
        if len(selected)<a.shape[0]:continue
        part=a[:,selected];coverage=float(np.linalg.eigvalsh(part@part.conj().T)[0])
        if coverage+1e-10>=minimum_coverage:
            return np.array(sorted(selected),dtype=int),dict(minimum_target_coverage=coverage,
                required_target_coverage=minimum_coverage,target_rank=a.shape[0],retained_roots=len(selected),
                fragmented=len(selected)>a.shape[0],complete_target_coverage=float(full[0]))
    raise ValueError(f'unresolved target subspace: coverage {coverage}, required {minimum_coverage}')
