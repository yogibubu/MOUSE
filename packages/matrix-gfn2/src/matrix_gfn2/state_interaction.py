"""Spin-adapted nonorthogonal state-interaction algebra.

These portable primitives require NumPy/SciPy, no external electronic program.
They accept prepared CSFs and matrix derivatives. Molecular orbital response
must be supplied by the corresponding ground/excited-state response solver;
``eigenvalue_jet`` alone does not construct a molecular Hessian.

A CSF is a sequence of (weight, (alpha_occupied, beta_occupied)) terms in an
orthonormal AO frame. The element implementation remains valid when occupied
bra/ket overlaps are singular, including orthogonal configurations.
"""
from ._nosi_slater import Hamiltonian
from ._nosi_mixed import element as determinant_element_mixed
from ._nosi_spin import csfs as four_open_shell_singlets
from ._nosi_spin import matrix as four_open_shell_spin_matrix
from ._nosi_eigen import Jet as MatrixDerivatives
from ._nosi_eigen import eigenvalue_jet, project_matrices

__all__ = [
    'Hamiltonian', 'determinant_element_mixed',
    'four_open_shell_singlets', 'four_open_shell_spin_matrix',
    'MatrixDerivatives', 'eigenvalue_jet', 'project_matrices',
]
