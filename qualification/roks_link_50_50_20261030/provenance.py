"""Freeze the LINK integration sources and completed Oracle evidence."""
from pathlib import Path
import hashlib,json,socket,sys
assert socket.getfqdn()=='oracle.sns.it'
BASE=Path('/home/enzo/MOUSE-roks-link50-20261030');OLD=Path('/home/enzo/MOUSE-roks-50-50-20261030')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files=[p for p in (BASE/'results').rglob('*') if p.is_file()]
files+=list((BASE/'source/qualification').glob('*.py'))
files+=[BASE/'source/packages/matrix-link/src/matrix_link'/n for n in ['scan.py','optimizer.py','qmqm_component.py','deltascf_component.py','composite_roks_component.py']]
files+=[BASE/'source/packages/matrix-gfn2/src/matrix_gfn2'/n for n in ['nonlocal_response.py','composite_roks.py','deltascf.py','spin_reference.py','deltascf_analytic_hessian.py','roks.py']]
files+=[BASE/'REPORT.md']
external={}
for name in ['nitroxyl','acetaldehyde']:
    for profile in ['full_response','qualified_spectral_response']:
        for leaf in ['response.npz','central.npz','analytic.npz']:
            p=OLD/'results'/name/profile/leaf;external[str(p)]=sha(p)
    p=OLD/'results'/name/'composite.npz';external[str(p)]=sha(p)
record=dict(host=socket.getfqdn(),python=sys.version,external_inputs_sha256=external,
    artifacts={str(p.relative_to(BASE)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(set(files))},
    tests_passed=44,molecular_integration=['nitroxyl','acetaldehyde'],
    revision_scope='final hardening validates response bytes before cache hits and rejects acceptance of another Hamiltonian; completed molecular derivative data retained, final persistence guards independently tested without repeating completed electronic jobs')
(BASE/'provenance.json').write_text(json.dumps(record,indent=2)+'\n');print('frozen artifacts',len(record['artifacts']))
