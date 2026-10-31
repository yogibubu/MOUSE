"""Isolated Oracle snapshot for the 30 October variational repair campaign."""
import importlib.util,os,socket,sys
from pathlib import Path
HOST=socket.getfqdn()
if HOST!='oracle.sns.it':raise RuntimeError(f'not an authorized compute host: {HOST}')
BASE=Path('/home/enzo/MOUSE-roks-independent-20261030')
for path in sorted((BASE/'source/packages').glob('*/src')):sys.path.insert(0,str(path))
native_path=next(Path('/home/enzo/nano-MATRIX-current/packages/matrix-gfn2/src/matrix_gfn2').glob('_gfn2_native*.so'))
spec=importlib.util.spec_from_file_location('matrix_gfn2._gfn2_native',native_path)
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
sys.modules[spec.name]=native
