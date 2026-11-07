"""Analytic metric and frame jets for positive density transport.

For reference Lowdin operators A, B(R)=sqrt(S) U A U.T sqrt(S).
This congruence transports positive AO densities to positive reference
densities. First/second square-root derivatives solve Sylvester equations;
there are no finite differences or eigenvector derivatives.
"""
import numpy as np
from .nonlocal_response import TransportedOperators


class LowdinOperatorJet:
    def __init__(self, state, anchor_coordinates_bohr, lowdin_operators, *,
                 overlap_first=None, overlap_second=None):
        self.transport = TransportedOperators(state, anchor_coordinates_bohr, lowdin_operators)
        values, vectors = np.linalg.eigh(state.overlap)
        if not np.all(np.isfinite(values)) or values.min() <= 1e-10:
            raise ValueError('positive density transport requires a nonsingular AO metric')
        self.vectors = vectors
        self.root_values = np.sqrt(values)
        self.root = (vectors*self.root_values)@vectors.T
        self.operators = self.root@self.transport.a@self.root
        nr, n = state.coordinates_bohr.size, len(values)
        self.root_first = self.root_second = None
        if overlap_first is not None:
            ds = np.asarray(overlap_first)
            if ds.shape != (nr,n,n) or not np.all(np.isfinite(ds)):
                raise ValueError('invalid first overlap jet')
            self.root_first = self._solve(ds)
        if overlap_second is not None:
            d2s = np.asarray(overlap_second)
            if self.root_first is None or d2s.shape != (nr,nr,n,n) or not np.all(np.isfinite(d2s)):
                raise ValueError('invalid second overlap jet')
            cross = self.root_first[:,None]@self.root_first[None,:]
            self.root_second = self._solve(d2s-cross-cross.swapaxes(0,1))

    def _solve(self, rhs):
        v = self.vectors
        denominator = self.root_values[:,None]+self.root_values[None,:]
        return v@((v.T@rhs@v)/denominator)@v.T

    def first(self):
        if self.root_first is None:
            raise ValueError('first overlap derivatives were not provided')
        l, dl, a = self.root, self.root_first, self.transport.a
        da = self.transport.first()
        return dl[:,None]@a@l+l@da@l+l@a@dl[:,None]

    def second(self, i, j):
        if self.root_second is None:
            raise ValueError('second overlap derivatives were not provided')
        l, a = self.root, self.transport.a
        li, lj, lij = self.root_first[i], self.root_first[j], self.root_second[i,j]
        da = self.transport.first()
        ai, aj, aij = da[i], da[j], self.transport.second(i,j)
        return (lij@a@l+li@aj@l+li@a@lj
                +lj@ai@l+l@aij@l+l@ai@lj
                +lj@a@li+l@aj@li+l@a@lij)

    def metric_potential(self, spin_densities, moment_fields):
        """Adjoint dE/dS for a differentiable energy of transported moments."""
        fields = np.asarray(moment_fields)
        densities = np.asarray(spin_densities)
        if fields.shape != (len(densities),len(self.transport.a)):
            raise ValueError('moment fields do not match the density channels')
        potentials = np.einsum('sa,aij->sij',fields,self.transport.a)
        rhs = sum(a@self.root@p+p@self.root@a for a,p in zip(potentials,densities))
        return self._solve(rhs)
