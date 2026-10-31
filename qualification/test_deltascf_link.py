"""LINK configuration and accepted/trial history contracts, no electronic jobs."""
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
import pytest
from matrix_link.scan import QMScanBackend
from matrix_link.qmqm_component import CappedQMComponent


def spec(**options):
 return QMScanBackend(name='matrix_gfn2',electronic_state=1,native_excited_method='roks',deltascf_promotion=(0,1),**options)


def test_explicit_configuration_and_old_tda_default():
 assert QMScanBackend(name='matrix_gfn2').native_excited_method=='tda'
 assert spec().deltascf_promotion==(0,1)
 assert replace(spec(),orbital_saddle_index=1).orbital_saddle_index==1
 with pytest.raises(ValueError):replace(spec(),orbital_saddle_index=-1)
 for changes in [dict(electronic_state=0),dict(deltascf_promotion=None),dict(name='gaussian'),dict(state_tracking='none'),dict(deltascf_promotion=(1,1)),dict(deltascf_promotion=(-1,2)),dict(multiplicity=3),dict(tda_virtual_count=2)]:
  with pytest.raises(ValueError):replace(spec(),**changes)


def test_trial_points_do_not_advance_accepted_orbital_state(monkeypatch,tmp_path):
 from matrix_link import deltascf_component as adapter
 calls=[]
 def fake(component,coordinates,compute_hessian,ground,diagnostics):
  calls.append(component.reference)
  return SimpleNamespace(scf_state=(object(),SimpleNamespace(state_id=1)),execution={'selected_electronic_state':1,'gradient_electronic_state':1})
 monkeypatch.setattr(adapter,'evaluate_component',fake)
 c=CappedQMComponent(spec(),SimpleNamespace(),tmp_path)
 first=c._native_excited(np.zeros((2,3)),False,None,{})
 second=c._native_excited(np.ones((2,3)),False,None,{})
 assert calls==[None,None] and c.reference is None
 c.accept(first)
 c._native_excited(np.ones((2,3)),False,None,{})
 assert calls[-1] is first.scf_state and c.reference is first.scf_state
 assert c.anchor is first.scf_state
