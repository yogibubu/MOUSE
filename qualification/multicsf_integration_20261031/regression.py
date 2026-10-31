"""Serialized Oracle regression of core, LINK and coherent APOC tracking."""
from pathlib import Path
import os,sys,socket,fcntl
assert socket.getfqdn()=='oracle.sns.it'
V=Path('/home/enzo/MOUSE-orbital-performance-20261031');W=V/'multicsf_integration'
os.environ.update(MOUSE_VALIDATION_ROOT=str(W),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2');sys.path.insert(0,str(W/'source/qualification'));import remote_bootstrap
from threadpoolctl import threadpool_limits
import pytest
files=['test_orbital_saddle.py','test_roks.py','test_nonlocal_response.py','test_relative_pair_roks.py','test_roks_trial_guard.py','test_composite_roks_link.py','test_deltascf_link.py','test_multicsf_apoc.py','test_multicsf_link.py']
if len(sys.argv)>1:files=sys.argv[1:]
with threadpool_limits(limits=2):
 lock=(V.parent/'MOUSE-publication-validation-20261030/blind/references/compute.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
 sys.exit(pytest.main(['-q',*(str(W/'source/qualification'/f) for f in files),'--junitxml='+str(W/'regression.xml')]))
