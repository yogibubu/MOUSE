"""Projector continuation without replacing physical eigenvectors by mixtures."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _clusters(values, tolerance):
    order = np.argsort(values, kind="stable")
    groups = []
    for i in order:
        if not groups or values[i] - values[groups[-1][0]] > tolerance:
            groups.append([int(i)])
        else:
            groups[-1].append(int(i))
    return groups


def transport_spectral_subspaces(reference_values, current_values, cross_gram, *,
                                  gap_tolerance=1.e-6, diagnostic_indices=None,
                                  reference_features=None, current_features=None,
                                  feature_weight=0.0):
    """Match projector spans, then eigenstates inside each matched span.

    Energies and gap_tolerance have the same units (MHz in LOCK). The cross
    Gram may be rectangular and include projection loss. No primitive-grid
    wavefunctions are needed. Polar rotations describe a transported *basis*,
    never replacement eigenvectors for an energy or intensity calculation.
    Feature columns may be transition-strength fingerprints in a common,
    explicitly supplied channel basis; only within-span assignment uses them.
    """
    left, right = np.asarray(reference_values, float), np.asarray(current_values, float)
    cross = np.asarray(cross_gram, complex)
    if (left.ndim != 1 or right.ndim != 1 or not len(left) or not len(right)
        or cross.shape != (len(left), len(right))
        or not all(np.all(np.isfinite(a)) for a in (left, right, cross))
        or not np.isfinite(gap_tolerance) or gap_tolerance < 0
        or not np.isfinite(feature_weight) or feature_weight < 0):
        raise ValueError("invalid spectral-subspace continuation inputs")
    selected = set(range(len(left))) if diagnostic_indices is None else set(map(int, diagnostic_indices))
    if selected - set(range(len(left))):
        raise ValueError("diagnostic state outside the reference spectrum")
    overlap = abs(cross)**2
    rows, cols = linear_sum_assignment(-overlap)
    mapping = dict(zip(map(int, rows), map(int, cols), strict=True))
    if selected - mapping.keys():
        raise ValueError("assigned state left the retained spectral space")
    # Unite energy clusters on both sides. This preserves the common span
    # across a split or merged near-degeneracy rather than following a vector.
    parent = np.arange(len(left))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return int(i)

    def unite(group):
        for i in group[1:]:
            parent[root(i)] = root(group[0])

    for group in _clusters(left, gap_tolerance):
        unite(group)
    inverse = {v: k for k, v in mapping.items()}
    right_clusters=_clusters(right,gap_tolerance)
    for group in right_clusters:
        unite([inverse[j] for j in group if j in inverse])
    groups = {}
    for i in range(len(left)):
        groups.setdefault(root(i), []).append(i)
    # Projector-overlap assignment with repeated rows enforces each span's
    # capacity. Internal basis rotations do not affect this first-stage score.
    score = overlap.copy()
    for group in groups.values():
        score[group] = overlap[group].sum(axis=0) / len(group)
    rows, cols = linear_sum_assignment(-score)
    assignment = dict(zip(map(int, rows), map(int, cols), strict=True))
    if selected - assignment.keys():
        raise ValueError("assigned subspace left the retained spectral space")
    feature_score = np.zeros_like(overlap)
    if reference_features is not None or current_features is not None:
        f, g = np.asarray(reference_features, float), np.asarray(current_features, float)
        if (f.ndim != 2 or g.shape != (len(right), f.shape[1]) or len(f) != len(left)
            or not np.all(np.isfinite(f)) or not np.all(np.isfinite(g))):
            raise ValueError("transition fingerprints must share finite feature channels")
        f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), np.finfo(float).tiny)
        g = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), np.finfo(float).tiny)
        feature_score = f @ g.T
    records, permutation = [], {}
    minimum = minimum_state = 1.0
    for group in groups.values():
        available = [i for i in group if i in assignment]
        targets = [assignment[i] for i in available]
        if selected.intersection(group) and len(available)!=len(group):
            raise ValueError("assigned quasi-degenerate span is incomplete; enlarge the retained spectrum")
        if selected.intersection(group):
            for current_group in right_clusters:
                if set(targets).intersection(current_group) and not set(current_group).issubset(targets):
                    raise ValueError("reference span omits a quasi-degenerate partner; enlarge the reference spectrum")
        local = cross[np.ix_(available, targets)]
        u, singular, vh = np.linalg.svd(local, full_matrices=False)
        local_score = overlap[np.ix_(available, targets)] + feature_weight * feature_score[np.ix_(available, targets)]
        rr, cc = linear_sum_assignment(-local_score)
        permutation.update({available[int(i)]: targets[int(j)] for i, j in zip(rr, cc, strict=True)})
        if selected.intersection(group):
            minimum = min(minimum, float(np.min(singular, initial=1.0)))
            minimum_state = min(minimum_state, min(overlap[i, permutation[i]] for i in available))
        polar = vh.conj().T @ u.conj().T
        # In the transported basis H is generally not diagonal. Store it so
        # a consumer cannot mistake quasi-degenerate mixtures for eigenstates.
        h = polar.conj().T @ np.diag(right[targets]) @ polar
        records.append(dict(reference_indices=available, current_indices=targets,
            singular_values=singular.tolist(),
            polar_rotation=dict(real=polar.real.tolist(), imag=polar.imag.tolist()),
            transported_hamiltonian=dict(real=h.real.tolist(), imag=h.imag.tolist()),
            individually_resolved=bool(len(group) == 1)))
    return dict(schema="matrix.lock.subspace_transport.v1", mapping=permutation,
                minimum_principal_overlap=minimum, minimum_state_overlap=minimum_state,
                subspaces=records, gap_tolerance=gap_tolerance,
                scope="projector transport; individual labels inside degenerate spans are gauge dependent")
