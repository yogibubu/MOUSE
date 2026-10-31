import socket,sys
from pathlib import Path
assert socket.getfqdn()=='oracle.sns.it'
sys.path.insert(0,'/home/enzo/MOUSE-roks-independent-20261030');import remote_bootstrap
BASE=Path('/home/enzo/MOUSE-roks-link50-20261030')
for p in (BASE/'source/packages').glob('*/src'):sys.path.insert(0,str(p))
import pytest
names=['test_composite_roks_link.py','test_nonlocal_response.py','test_deltascf_link.py','test_anchored_roks_link.py','test_tensor_response.py','test_frozen_response_reference.py','test_roks.py','test_deltascf.py','test_orbital_saddle.py','test_energy_only_checkpoint.py']
raise SystemExit(pytest.main([str(BASE/'source/qualification'/n) for n in names]+['-q']))
