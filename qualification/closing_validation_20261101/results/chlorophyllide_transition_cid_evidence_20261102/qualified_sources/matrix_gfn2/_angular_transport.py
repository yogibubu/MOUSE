"""Real d-shell rotations in native (z2,xz,yz,x2-y2,xy) order."""
import numpy as np

_D = np.zeros((5, 3, 3))
_D[0] = np.diag([-1., -1., 2.]) / np.sqrt(6.)
_D[1, 0, 2] = _D[1, 2, 0] = 1 / np.sqrt(2.)
_D[2, 1, 2] = _D[2, 2, 1] = 1 / np.sqrt(2.)
_D[3] = np.diag([1., -1., 0.]) / np.sqrt(2.)
_D[4, 0, 1] = _D[4, 1, 0] = 1 / np.sqrt(2.)


def d_bilinear(left, right):
    return np.einsum('aij,...jk,bkl,...il->...ab', _D, left, _D, right,
                     optimize=True)


def d_rotation(rotation):
    return d_bilinear(rotation, rotation)


def d_first(rotation, direction):
    return d_bilinear(direction, rotation) + d_bilinear(rotation, direction)


def d_second(rotation, first, second, mixed):
    return (d_first(rotation, mixed) + d_bilinear(first, second)
            + d_bilinear(second, first))
