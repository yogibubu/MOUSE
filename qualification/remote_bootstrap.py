"""Explicit Oracle validation environment; never a local calculation fallback."""
import importlib.util
import os
from pathlib import Path
import socket
import sys

HOST = socket.getfqdn()
if HOST != 'oracle.sns.it':
    raise RuntimeError(f'remote compute-host identity check failed: {HOST}')
BASE = Path(os.environ.get('MOUSE_VALIDATION_ROOT','/home/enzo/MOUSE-deltascf-validation-20261029'))
for path in sorted((BASE/'source/packages').glob('*/src')):
    sys.path.insert(0,str(path))
# Reuse the compatible, already compiled Linux extension; no laptop binary transfer.
native_path = next(Path('/home/enzo/nano-MATRIX-current/packages/matrix-gfn2/src/matrix_gfn2').glob('_gfn2_native*.so'))
spec = importlib.util.spec_from_file_location('matrix_gfn2._gfn2_native',native_path)
native = importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
sys.modules[spec.name] = native
