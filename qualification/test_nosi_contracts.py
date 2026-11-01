"""NOSI identity, legacy restart compatibility and tamper rejection."""
from dataclasses import asdict,replace
from pathlib import Path
import shutil
import pytest
from matrix_gfn2.nosi_surface import PreparedNOSIModel,load_nosi_result
from matrix_link.scan import QMScanBackend,_composite_restart_json_value
from matrix_link.optimizer import _restart_identity_value

FIXTURE=Path(__file__).parent/'response_consistent_nosi_20261101/results/prepared_model_criegee'

def test_selector_identity_and_legacy_contract(tmp_path):
    p=tmp_path/'model.json';p.write_text('{}')
    b=QMScanBackend(name='matrix_gfn2',electronic_state=1,native_excited_method='nosi_50_50',native_nosi_model_path=p)
    assert QMScanBackend(**asdict(b))==b
    for encode in [_composite_restart_json_value,_restart_identity_value]:
        assert 'native_nosi_model_path' not in encode(QMScanBackend(name='matrix_gfn2'))
        assert encode(b)['native_nosi_model_sha256']==b.native_nosi_model_sha256
    with pytest.raises(ValueError):replace(b,native_excited_method='tda')
    with pytest.raises(ValueError):replace(b,matrix_gfn2_alpb_solvent='water')
    p.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='checksum'):QMScanBackend(**asdict(b))


def test_full_numeric_bundle_rejects_mutation(tmp_path):
    shutil.copytree(FIXTURE,tmp_path/'model');m=PreparedNOSIModel(tmp_path/'model/model.json')
    with (m.root/'arrays.npz').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='checksum'):m.validate_files()
    with pytest.raises(ValueError,match='checksum'):m.evaluate(None,m.anchor.coordinates_bohr,properties=('energy',))
