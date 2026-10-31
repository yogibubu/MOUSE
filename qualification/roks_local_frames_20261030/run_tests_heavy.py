import socket,sys,os
from pathlib import Path
assert socket.getfqdn()=='oracle.sns.it'
BASE=Path('/home/enzo/MOUSE-roks-heavy-frames-20261030');os.environ['MOUSE_VALIDATION_ROOT']=str(BASE)
sys.path.insert(0,'/home/enzo/MOUSE-roks-independent-20261030');import remote_bootstrap
for path in (BASE/'source/packages').glob('*/src'):sys.path.insert(0,str(path))
# Load the frozen source snapshot before archive fixtures locate their data.
import matrix_gfn2,matrix_link.scan,matrix_link.qmqm_component
import pytest
names=['test_local_frame_roks.py','test_roks_geometry_only.py','test_response_reference_homotopy.py','test_roks_tracking.py','test_composite_roks_link.py','test_nonlocal_response.py','test_deltascf_link.py','test_anchored_roks_link.py','test_tensor_response.py','test_frozen_response_reference.py','test_roks.py','test_deltascf.py','test_orbital_saddle.py','test_energy_only_checkpoint.py']
raise SystemExit(pytest.main([str(BASE/'source/qualification'/n) for n in names]+['-q']))
