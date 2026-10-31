"""Public selector contracts, cache validation and analytic final Hessian route."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from matrix_link.scan import QMScanBackend,_composite_restart_json_value

def backend(tmp_path):
 path=tmp_path/'model.json';path.write_text('{"schema":"mouse.multicsf.model.v1"}\n')
 return QMScanBackend(name='matrix_gfn2',electronic_state=1,native_excited_method='multicsf_50_50',native_multicsf_model_path=path)

def test_multicsf_selector_has_its_own_frozen_model_contract(tmp_path):
 from matrix_link.optimizer import _restart_identity_value,_electronic_restart_artifact_identity
 spec=backend(tmp_path);assert spec.deltascf_promotion is None
 assert _electronic_restart_artifact_identity(spec)['native_multicsf_model_path']['sha256']==spec.native_multicsf_model_sha256
 for serializer in [_restart_identity_value,_composite_restart_json_value]:
  assert 'native_multicsf_model_path' not in serializer(QMScanBackend(name='matrix_gfn2'))
  assert serializer(spec)['native_multicsf_model_sha256']==spec.native_multicsf_model_sha256
 for changes in [dict(native_multicsf_model_path=None),dict(deltascf_promotion=(0,1)),dict(native_excited_method='tda'),dict(native_multicsf_model_sha256='changed'),dict(matrix_gfn2_alpb_solvent='water')]:
  with pytest.raises(ValueError):replace(spec,**changes)

def test_multicsf_checks_prepared_inputs_before_optimizer_cache(tmp_path):
 from matrix_link.optimizer import GeometryEvaluationService
 spec=backend(tmp_path);service=object.__new__(GeometryEvaluationService)
 def changed():raise ValueError('input changed')
 service._orbital_component=SimpleNamespace(backend=spec,_multicsf_model=SimpleNamespace(validate_files=changed))
 with pytest.raises(ValueError,match='input changed'):service.evaluate(np.zeros(3),tag='cached')

def test_multicsf_final_hessian_uses_analytic_component(monkeypatch,tmp_path):
 import matrix_link.optimizer as module
 spec=backend(tmp_path);calls=[];answer=SimpleNamespace(result=SimpleNamespace(hessian_hartree_per_bohr2=np.eye(3)))
 service=SimpleNamespace(backend=spec,evaluate=lambda q,**kw:(calls.append(kw) or answer))
 def forbidden(*a,**kw):raise AssertionError('analytic multi-CSF Hessian fell back to finite differences')
 monkeypatch.setattr(module,'_finite_difference_final_cartesian_hessian',forbidden)
 assert module._evaluate_final_cartesian_hessian(service,None,np.zeros(3),tag='final') is answer
 assert calls[0]['requested_properties']==('energy','gradient','hessian')
