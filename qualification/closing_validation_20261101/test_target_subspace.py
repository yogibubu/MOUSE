"""Physical target fragmentation and reference-frame invariance."""
import numpy as np
from matrix_gfn2._nosi_target_subspace import select_target_subspace


def test_fragmented_one_target_keeps_both_mixed_roots():
    a=np.sqrt([[.001,.42,.385,.10]])
    roots,report=select_target_subspace(a)
    assert roots.tolist()==[1,2]
    assert report['fragmented'] and abs(report['minimum_target_coverage']-.805)<1e-14


def test_coverage_protects_each_target_and_is_unitary_invariant():
    a=np.array([[.8,.5,0,0],[0,0,.65,.6]])
    roots,report=select_target_subspace(a)
    theta=.71;u=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    rotated,other=select_target_subspace(u@a)
    assert roots.tolist()==[0,2,3] and rotated.tolist()==roots.tolist()
    assert abs(report['minimum_target_coverage']-.64)<1e-14
    assert abs(other['minimum_target_coverage']-report['minimum_target_coverage'])<1e-14
