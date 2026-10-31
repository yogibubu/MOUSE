from common import *
from science_slots import science_locks
import pytest
locks=science_locks()
with threadpool_limits():
 code=pytest.main(['-q',str(OUT/'test_surface_contracts.py'),'--junitxml='+str(OUT/'surface_contracts.xml')]);raise SystemExit(code)
