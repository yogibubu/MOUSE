import sys
sys.path.insert(0,'/home/enzo/MOUSE-roks-independent-20261030')
import remote_bootstrap
from pathlib import Path
for p in Path('/home/enzo/MOUSE-roks-50-50-20261030/source/packages').glob('*/src'):sys.path.insert(0,str(p))
import pytest
tests=['test_nonlocal_response.py','test_tensor_response.py','test_frozen_response_reference.py','test_roks.py','test_deltascf.py','test_orbital_saddle.py','test_deltascf_link.py','test_anchored_roks_link.py','test_energy_only_checkpoint.py']
raise SystemExit(pytest.main([str(Path('/home/enzo/MOUSE-roks-50-50-20261030/source/qualification')/n) for n in tests]+['-q']))
