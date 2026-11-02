"""Collinear on-site shell spin polarization with published spGFN constants.

E_spin = 1/2 m.T W m, with Mulliken shell magnetizations m = n_alpha-n_beta.
The two Fock contributions are opposite derivatives of this same energy.
Constants retain their upstream LGPL license and pinned provenance in data/.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from importlib.resources import files
import tomllib

import numpy as np

SPIN_PARAMETER_SHA256 = 'f2bd4c7281f7599d21e3784297c2575037f506526dfa47b562affa31dd32f0a2'


@lru_cache(maxsize=1)
def spin_constants():
    raw = files('matrix_gfn2').joinpath('data/spin_constants.toml').read_bytes()
    if sha256(raw).hexdigest() != SPIN_PARAMETER_SHA256:
        raise ValueError('spin polarization parameter checksum mismatch')
    table = tomllib.loads(raw.decode())
    if table['meta']['unit'] != 'hartree':
        raise ValueError('spin constants must be expressed in hartree')
    # Return immutable numerical tuples, not a mutable cached table.
    return tuple(tuple(table['elements'][str(z)]) for z in range(1, 87))


@dataclass(frozen=True)
class ShellSpinPolarization:
    overlap: np.ndarray
    orbital_shell: np.ndarray
    coupling_hartree: np.ndarray

    def __post_init__(self):
        s = np.array(self.overlap, dtype=float, copy=True)
        shells = np.array(self.orbital_shell, dtype=int, copy=True)
        w = np.array(self.coupling_hartree, dtype=float, copy=True)
        if (s.ndim != 2 or s.shape[0] != s.shape[1] or shells.shape != (len(s),)
                or w.ndim != 2 or w.shape[0] != w.shape[1]
                or np.any(shells < 0) or np.any(shells >= len(w))
                or not np.all(np.isfinite(s)) or not np.all(np.isfinite(w))
                or not np.allclose(s, s.T, atol=1e-12, rtol=0)
                or not np.allclose(w, w.T, atol=1e-12, rtol=0)):
            raise ValueError('invalid shell spin polarization metric or coupling')
        for name, array in [('overlap', s), ('orbital_shell', shells), ('coupling_hartree', w)]:
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @classmethod
    def from_gfn2(cls, state, partition):
        shells = np.asarray(partition.orbital_shell, dtype=int)
        shell_atoms = np.asarray(partition.shell_atom, dtype=int)
        angular = []
        for shell, atom in enumerate(shell_atoms):
            indices = np.flatnonzero(shells == shell)
            if not len(indices) or np.any(np.asarray(state.ao_atom)[indices] != atom):
                raise ValueError('GFN2 shell/atom metadata are inconsistent')
            labels = {state.ao_shell[i][-1].lower() for i in indices}
            if len(labels) != 1 or not labels <= set('spd'):
                raise ValueError('spin constants support explicit s/p/d shells only')
            angular.append('spd'.index(labels.pop()))
        constants = spin_constants()
        pairs = np.array([[0, 1, 3], [1, 2, 4], [3, 4, 5]])
        w = np.zeros((len(shell_atoms), len(shell_atoms)))
        for i, atom in enumerate(shell_atoms):
            z = int(state.atomic_numbers[atom])
            if not 1 <= z <= len(constants):
                raise ValueError(f'no published spin constants for atomic number {z}')
            for j in np.flatnonzero(shell_atoms == atom):
                w[i, j] = constants[z-1][pairs[angular[i], angular[j]]]
        return cls(state.overlap, shells, w)

    def evaluate(self, spin_densities):
        p = np.asarray(spin_densities, dtype=float)
        if (p.shape != (2, *self.overlap.shape) or not np.all(np.isfinite(p))
                or not np.allclose(p, p.transpose(0, 2, 1), atol=1e-12, rtol=0)):
            raise ValueError('spin polarization requires symmetric alpha/beta AO densities')
        magnetization = np.bincount(self.orbital_shell,
            weights=np.sum((p[0]-p[1])*self.overlap, axis=1),
            minlength=len(self.coupling_hartree))
        potential = self.coupling_hartree @ magnetization
        ao = potential[self.orbital_shell]
        v = .5*self.overlap*(ao[:, None]+ao[None, :])
        return np.array([v, -v]), .5*float(magnetization @ potential), magnetization

    def response_batch(self, spin_densities):
        """Linear spin Fock response for multiple symmetric perturbations."""
        p=np.asarray(spin_densities,dtype=float)
        if p.ndim!=4 or p.shape[1:]!=(2,*self.overlap.shape):
            raise ValueError('invalid batch of spin density perturbations')
        populations=np.sum((p[:,0]-p[:,1])*self.overlap,axis=2)
        shell_map=np.eye(len(self.coupling_hartree))[self.orbital_shell]
        potential=(populations@shell_map)@self.coupling_hartree.T
        ao=potential[:,self.orbital_shell]
        v=.5*self.overlap*(ao[:,:,None]+ao[:,None,:])
        return np.stack((v,-v),axis=1)
