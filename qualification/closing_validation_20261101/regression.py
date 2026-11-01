"""Run meaningful spin, algebra and legacy-contract regressions remotely."""
import socket,runpy,sys
from pathlib import Path
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
base=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/source')
runpy.run_path(str(base/'qualification/remote_bootstrap.py'))
import pytest
raise SystemExit(pytest.main(['-q',str(base/'qualification/test_screened_cid.py'),str(base/'qualification/test_nosi_contracts.py')]))
