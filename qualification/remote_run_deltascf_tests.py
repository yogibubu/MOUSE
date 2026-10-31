from remote_bootstrap import BASE,HOST
import json,hashlib,platform,sys
import numpy,scipy,pytest
source=BASE/'source'
files=list((source/'packages/matrix-gfn2/src/matrix_gfn2').glob('deltascf*.py'))
files += [source/'packages/matrix-gfn2/src/matrix_gfn2'/name for name in ('roks.py','spin_polarization.py')]
environment=dict(host=HOST,python=sys.version,numpy=numpy.__version__,scipy=scipy.__version__,
    machine=platform.machine(),source_files={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    native_extension_reused='/home/enzo/nano-MATRIX-current/packages/matrix-gfn2/src/matrix_gfn2/_gfn2_native.cpython-311-x86_64-linux-gnu.so',
    compute_threads=2)
(BASE/'results/environment.json').write_text(json.dumps(environment,indent=2)+'\n')
raise SystemExit(pytest.main([str(source/'qualification/test_deltascf.py'),str(source/'qualification/test_roks.py'),'-q']))
