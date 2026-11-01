"""Matched native-orbital TDA for the four prospective extension cases."""
import socket,runpy,importlib.util
from pathlib import Path
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
runpy.run_path('/Users/vincenzobarone/MOUSE-oo-response-20261101/source/qualification/remote_bootstrap.py')
p=Path('/Users/vincenzobarone/MOUSE-oo-response-20261101/oracle/MOUSE-frontier-cisd-20261101/tda_control.py')
spec=importlib.util.spec_from_file_location('matched_tda_control',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
from threadpoolctl import threadpool_limits
with threadpool_limits(2):
 for cohort,name in [('post_kernel_validation','tetrazine'),('post_kernel_validation','acrolein'),('cid_prospective','nitrosomethane'),('cid_prospective','maleimide')]:
  module.run(cohort,name);print(cohort,name,flush=True)
