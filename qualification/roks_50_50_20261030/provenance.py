"""Freeze final sources, numerical checkpoints and external input identities."""
from pathlib import Path
import hashlib,json,socket,platform,sys
BASE=Path('/home/enzo/MOUSE-roks-50-50-20261030');OLD=Path('/home/enzo/MOUSE-roks-independent-20261030')
assert socket.getfqdn()=='oracle.sns.it'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
paths=list((BASE/'results').rglob('*'))+list((BASE/'source/qualification').glob('*.py'))
for name in ['nonlocal_response.py','composite_roks.py','deltascf.py','spin_reference.py','__init__.py','deltascf_analytic_hessian.py','roks.py','roks_orbital_response.py']:
    paths.append(BASE/'source/packages/matrix-gfn2/src/matrix_gfn2'/name)
paths += [BASE/'REPORT.md',BASE/'run_protocol.json']
external={}
for name in ['nitroxyl','acetaldehyde']:
    for profile in ['full_response','qualified_spectral_response']:
        folder=OLD/'results'/profile/name
        for p in [folder/'model.npz',folder/'n_pi/result.json']:
            external[str(p)]=sha(p)
        row=json.loads((folder/'n_pi/result.json').read_text());external[row['checkpoint']]=sha(Path(row['checkpoint']))
external[str(OLD/'results/edft_provenance.json')]=sha(OLD/'results/edft_provenance.json')
data=dict(host=socket.getfqdn(),python=sys.version,platform=platform.platform(),
    source_scope='isolated snapshot; no previous campaign sources modified',external_inputs_sha256=external,
    artifacts={str(p.relative_to(BASE)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(set(paths)) if p.is_file()},
    nuclear_derivatives='analytic; finite differences only for independent validation',
    revisions='initial rotation diagnostic reused a mutable calculator across geometries; fixed diagnostic with separate calculators. Composite overlap explicitly compares different Hamiltonians, preserving APOC trajectory guards. Completed electronic points unchanged and resumed.')
(BASE/'provenance.json').write_text(json.dumps(data,indent=2)+'\n');print('frozen artifacts',len(data['artifacts']))
