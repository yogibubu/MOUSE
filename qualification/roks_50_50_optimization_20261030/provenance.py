"""Freeze the completed optimization campaign and its negative qualification."""
from pathlib import Path
import socket,hashlib,json,sys
assert socket.getfqdn()=='oracle.sns.it'
BASE=Path('/home/enzo/MOUSE-roks-50-50-optimization-20261030')
OLD=Path('/home/enzo/MOUSE-roks-independent-20261030')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files=[p for root in ['results','references'] for p in (BASE/root).rglob('*') if p.is_file()]
files+=list((BASE/'source/qualification').glob('*.py'))
files+=[BASE/'source/packages/matrix-link/src/matrix_link'/n for n in ['optimizer.py','scan.py','composite_roks_component.py','deltascf_component.py','qmqm_component.py']]
files+=[BASE/'source/packages/matrix-gfn2/src/matrix_gfn2'/n for n in ['nonlocal_response.py','composite_roks.py','deltascf.py','deltascf_analytic_hessian.py','roks.py','roks_orbital_response.py','spin_reference.py']]
files+=[BASE/'protocol.json',BASE/'REPORT.md']
external={}
for name in ['nitroxyl','acetaldehyde','maleimide']:
 for profile in ['full_response','qualified_spectral_response']:
  for leaf in ['model.npz','tda.json','calibration.json']:
   p=OLD/'results'/profile/name/leaf
   if p.exists():external[str(p)]=sha(p)
for p in [Path('/home/enzo/MOUSE-roks-50-50-20261030/provenance.json'),Path('/home/enzo/MOUSE-roks-link50-20261030/provenance.json')]:
 if p.exists():external[str(p)]=sha(p)
record=dict(host=socket.getfqdn(),python=sys.version,weights=[.5,.5],lambda_exchange=.9,
    tests_passed=47,main_molecules=3,certified_minima=0,completed_symmetry_breaking_branches=4,
    all_compute_complete=True,scientific_status='not qualified for general excited-state geometries/frequencies under the frozen protocol',
    source_repairs=['analytic final composite Hessian dispatch','excited TDA numerical fallback preserves Hamiltonian','energy-only initial anchor restore'],
    invalidated_artifact='acetaldehyde/main/optimizer_result.json final positive frequencies came from the obsolete ground-state fallback; final.npz is the independent analytic composite result',
    external_inputs_sha256=external,
    artifacts={str(p.relative_to(BASE)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(set(files))})
(BASE/'provenance.json').write_text(json.dumps(record,indent=2)+'\n');print('frozen artifacts',len(record['artifacts']))
