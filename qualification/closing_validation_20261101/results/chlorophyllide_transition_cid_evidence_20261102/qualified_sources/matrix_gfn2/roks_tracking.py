"""Phase invariant active-orbital character in the frozen response AO frame.

This diagnostic deliberately excludes the closed-shell determinant product.
It is not a state overlap or a spectroscopic n/pi classifier; local full-CSF
continuity remains necessary when using it as a long-path character guard.
"""
import numpy as np
from .nonlocal_response import TransportedOperators


def assigned_transition_orbitals(coefficients, overlap, occupations, occupied,
                                virtual, transition, *, minimum_rank_one=.70,
                                minimum_margin=.05):
    """Seed a common-orbital CSF from the leading assigned NTO pair.

    Rotate within the retained ground's occupied and virtual spaces, placing
    the leading singular vectors at the CSF's hole and particle labels. The
    closed-shell ground density is unchanged. No excitation energy or guessed
    saddle index enters the initialization.
    """
    c,s,occ,x=(np.asarray(a,dtype=float) for a in
               (coefficients,overlap,occupations,transition))
    oi,vi=(np.asarray(a,dtype=int) for a in (occupied,virtual))
    n=len(s)
    if (s.shape!=(n,n) or c.shape!=(n,n) or occ.shape!=(2,n)
            or x.shape!=(len(oi),len(vi)) or not len(oi) or not len(vi)
            or not all(np.all(np.isfinite(a)) for a in (c,s,occ,x))
            or not np.all((occ==0)|(occ==1))
            or len(np.unique(np.r_[oi,vi]))!=n
            or not np.array_equal(np.sort(np.r_[oi,vi]),np.arange(n))
            or not np.allclose(c.T@s@c,np.eye(n),atol=1e-7,rtol=0)
            or not all(np.isfinite(t) and 0<=t<=1 for t in
                       (minimum_rank_one,minimum_margin))):
        raise ValueError('invalid assigned-transition orbital chart')
    hole=np.flatnonzero(occ[1]-occ[0]==1)
    particle=np.flatnonzero(occ[0]-occ[1]==1)
    if (len(hole)!=1 or len(particle)!=1 or hole[0] not in oi
            or particle[0] not in vi):
        raise ValueError('assigned CSF needs one occupied hole and virtual particle')
    u,sv,vt=np.linalg.svd(x,full_matrices=True)
    norm=np.linalg.norm(sv)
    margin=1-sv[1]/sv[0] if norm and len(sv)>1 else 1.
    if not norm or sv[0]/norm<minimum_rank_one or margin<minimum_margin:
        raise ValueError('assigned transition lacks a distinct single-pair direction')
    hi=int(np.flatnonzero(oi==hole[0])[0])
    pi=int(np.flatnonzero(vi==particle[0])[0])
    v=vt.T.copy()
    u[:,[0,hi]]=u[:,[hi,0]]
    v[:,[0,pi]]=v[:,[pi,0]]
    result=c.copy()
    result[:,oi]=c[:,oi]@u
    result[:,vi]=c[:,vi]@v
    return result


def common_orbital_singlet_overlap(left, right, overlap, occupations):
    """Absolute coherent CSF overlap at one fixed geometry, including spectators.

    Both determinants use the same spatial orbitals and alpha/beta-block
    ordering. The two diagonal and two spin-exchanged determinant terms reduce
    to two products. This is an overlap, not an energy projection.
    """
    c, d, s = map(np.asarray, (left, right, overlap))
    occ = np.asarray(occupations)
    if (s.ndim != 2 or s.shape[0] != s.shape[1] or c.shape != s.shape
            or d.shape != s.shape or occ.shape != (2, len(s))
            or not np.all((occ == 0) | (occ == 1))
            or np.count_nonzero(occ[0]-occ[1] == 1) != 1
            or np.count_nonzero(occ[1]-occ[0] == 1) != 1):
        raise ValueError('common-orbital singlet overlap needs two open orbitals')
    ia, ib = (np.flatnonzero(o) for o in occ)
    cross = c.T @ s @ d
    value = (np.linalg.det(cross[np.ix_(ia, ia)]) * np.linalg.det(cross[np.ix_(ib, ib)])
             + np.linalg.det(cross[np.ix_(ia, ib)]) * np.linalg.det(cross[np.ix_(ib, ia)]))
    if not np.isfinite(value) or abs(value) > 1+1e-5:
        raise ValueError('invalid normalized singlet overlap')
    return float(min(1., abs(value)))


def make_roks_trial_guard(overlap, occupations, anchor_coefficients, *,
                          minimum_local_overlap=.90, minimum_active_overlap=.70,
                          reference_coefficients=None, occupied=None, virtual=None,
                          reference_transition=None, minimum_transition_overlap=.70,
                          minimum_sector_norm=.70, transition_metric='raw',
                          minimum_rank_one_overlap=.70, minimum_rank_one_margin=.05):
    """Reject orbital steps that change state, without changing the objective.

    This fixed-geometry callback is suitable for GMF's ``trial_guard``. It
    combines local full-CSF continuity with a fixed active-pair anchor and,
    optionally, a previously assigned transition in fixed occupied/virtual
    spaces. Open-orbital exchange and independent phases are harmless.
    Stationarity and the full orbital index still require separate certificates.
    """
    s = np.array(overlap, copy=True)
    occ = np.array(occupations, copy=True)
    anchor = np.array(anchor_coefficients, copy=True)
    common_orbital_singlet_overlap(anchor, anchor, s, occ)
    thresholds = (minimum_local_overlap, minimum_active_overlap,
                  minimum_transition_overlap, minimum_sector_norm,
                  minimum_rank_one_overlap, minimum_rank_one_margin)
    if not all(np.isfinite(x) and 0 <= x <= 1 for x in thresholds):
        raise ValueError('state guard thresholds must lie in [0, 1]')
    if transition_metric not in {'raw', 'rank_one'}:
        raise ValueError('unknown assigned-transition overlap metric')
    opens = np.flatnonzero(occ[0] != occ[1])
    anchor_dual = anchor[:, opens].T @ s
    transition = None
    rank_one_limit = 1.
    rank_one_margin = 1.
    if reference_transition is not None:
        cref = np.asarray(reference_coefficients)
        oi, vi = np.asarray(occupied, dtype=int), np.asarray(virtual, dtype=int)
        transition = np.array(reference_transition, dtype=float, copy=True)
        if (cref.shape != s.shape or transition.shape != (len(oi), len(vi))
                or not np.all(np.isfinite(transition)) or np.linalg.norm(transition) < 1e-12):
            raise ValueError('invalid assigned transition for orbital state guard')
        transition /= np.linalg.norm(transition)
        singular = np.linalg.svd(transition, compute_uv=False)
        rank_one_limit = float(singular[0])
        rank_one_margin = float(1-singular[1]/singular[0]) if len(singular)>1 else 1.
        if transition_metric == 'rank_one' and (rank_one_limit < minimum_rank_one_overlap
                or rank_one_margin < minimum_rank_one_margin):
            raise ValueError('assigned transition lacks a distinct representable single-pair direction')
        occupied_dual, virtual_dual = cref[:, oi].T @ s, cref[:, vi].T @ s

    def guard(current, trial):
        local = common_orbital_singlet_overlap(current, trial, s, occ)
        cross = anchor_dual @ trial[:, opens]
        active = float(abs(cross[0, 0]*cross[1, 1]+cross[0, 1]*cross[1, 0]))
        character, sector = 1., 1.
        if transition is not None:
            candidates = []
            for h, p in (opens, opens[::-1]):
                hole, particle = occupied_dual @ trial[:, h], virtual_dual @ trial[:, p]
                rank = np.outer(hole, particle)
                norm = float(np.linalg.norm(rank))
                score = float(abs(np.sum(rank*transition))/norm) if norm > 1e-12 else 0.
                candidates.append((score, min(float(np.linalg.norm(hole)), float(np.linalg.norm(particle)))))
            admissible = [(q, n) for q, n in candidates if n >= minimum_sector_norm]
            character, sector = max(admissible or candidates)
        raw_character = character
        if transition is not None and transition_metric == 'rank_one':
            character = min(1., character/rank_one_limit)
        accepted = (local >= minimum_local_overlap and active >= minimum_active_overlap
                    and character >= minimum_transition_overlap and sector >= minimum_sector_norm)
        guard.last_scores = dict(local_csf=local, anchor_active=active,
                                 transition=character, raw_transition=raw_character,
                                 transition_metric=transition_metric,
                                 rank_one_limit=rank_one_limit, rank_one_margin=rank_one_margin,
                                 sector_norm=sector, accepted=accepted)
        guard.evaluations += 1
        guard.rejections += not accepted
        return accepted

    guard.evaluations, guard.rejections, guard.last_scores = 0, 0, None
    return guard


def active_orbital_character(anchor, candidate, frame_coordinates_bohr, *, allow_different_response=False):
    """Return open-orbital and symmetrized active-pair overlap amplitudes.

    Symmetric orthogonalization defines a fixed labelled AO gauge. Each vector
    is transported back into the same proper molecular frame, so translations,
    rigid rotations and independent MO phases do not affect the diagnostic.
    """
    a,b=anchor.reference_state,candidate.reference_state
    if a is None or b is None:
        raise ValueError('active-orbital character requires molecular metadata')
    for name in ('atomic_numbers','ao_atom','ao_shell'):
        if not np.array_equal(getattr(a,name),getattr(b,name)):
            raise ValueError('active-orbital character requires the same labelled AO basis')
    if (a.model_id!=b.model_id or a.charge!=b.charge
            or (not allow_different_response and anchor.spin_reference_fingerprint!=candidate.spin_reference_fingerprint)
            or not np.array_equal(anchor.core_electrons,candidate.core_electrons)
            or not np.array_equal(anchor.occupations,candidate.occupations)):
        raise ValueError('active-orbital character requires the same model and occupations')
    difference=anchor.occupations[0]-anchor.occupations[1]
    hole=np.flatnonzero(difference==-1);particle=np.flatnonzero(difference==1)
    if len(hole)!=1 or len(particle)!=1:
        raise ValueError('active-orbital character requires exactly two open orbitals')
    ids=[int(hole[0]),int(particle[0])];vectors=[]
    clusters=getattr(anchor,'spin_frame_clusters',None)
    other_clusters=getattr(candidate,'spin_frame_clusters',None)
    if (clusters is None)!=(other_clusters is None) or clusters is not None and not np.array_equal(clusters,other_clusters):
        raise ValueError('active-orbital character requires the same frozen frame topology')
    for result,state in ((anchor,a),(candidate,b)):
        eigen,v=np.linalg.eigh(result.overlap)
        if not np.all(np.isfinite(eigen)) or eigen.min()<=1e-10:
            raise ValueError('active-orbital AO metric is singular')
        root=(v*np.sqrt(eigen))@v.T
        if clusters is None:
            transport=TransportedOperators(state,frame_coordinates_bohr,np.eye(len(eigen))[None])
        else:
            from .local_frame_response import LocalTransportedOperators
            transport=LocalTransportedOperators(state,frame_coordinates_bohr,np.eye(len(eigen))[None],clusters)
        q=transport.u.T@root@result.common_coefficients[:,ids]
        if not np.allclose(q.T@q,np.eye(2),atol=1e-8,rtol=0):
            raise ValueError('active orbitals are not orthonormal')
        vectors.append(q)
    cross=vectors[0].T@vectors[1]
    # A singlet's open orbitals form an unordered pair: swapping them gives
    # the SAME CSF. Compare normalized symmetric dyads, preserving signs
    # until after summation, rather than rejecting a mere label exchange.
    score=abs(cross[0,0]*cross[1,1]+cross[0,1]*cross[1,0])
    return dict(hole=float(np.clip(abs(cross[0,0]),0.,1.)),
        particle=float(np.clip(abs(cross[1,1]),0.,1.)),
        transition=float(np.clip(score,0.,1.)))
