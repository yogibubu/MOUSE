"""Compact, checksummed candidate evidence; preserve large arrays on compute host."""
from pathlib import Path
import socket,hashlib,json,shutil
assert socket.getfqdn()=='mac-2.homenet.telecomitalia.it'
HERE=Path(__file__).resolve().parent;BASE=HERE.parents[1];out=HERE/'static_subtraction_evidence_20261102';out.mkdir(exist_ok=True)
def copy(source,destination):
 destination=out/destination;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
completion=json.loads((HERE/'static_subtraction_models/completion.json').read_text());assert len(completion)==24 and all(r['returncode']==0 for r in completion)
copy(HERE/'static_subtraction_models/completion.json','public_models/completion.json')
large=[]
for name in ['static_subtraction_models','static_subtraction_small_molecules']:
 for p in (HERE/name).rglob('*'):
  if not p.is_file():continue
  relative=p.relative_to(HERE)
  if p.suffix=='.json' or (p.name.startswith('hessian') and p.suffix=='.npz') or p.name=='runtime_checkpoint.npz' or (name=='static_subtraction_small_molecules' and p.suffix=='.npz'):
   copy(p,relative)
  elif p.suffix=='.npz':large.append(dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p),retained_on_remote=True))
for pattern in ['chlorophyllide_static_subtraction*.json','chlorophyllide_static_subtraction*.npz','beta_static_subtraction*.json','beta_static_subtraction*.npz','chlorophyllide_balanced_ci_diagnostic.*']:
 for p in HERE.glob(pattern):
  if p.is_file():copy(p,'pigments/'+p.name)
for p in (HERE/'static_subtraction_frontiers').rglob('*'):
 if p.is_file() and p.suffix in ('.json','.npz'):
  if p.stat().st_size<90*1024*1024:copy(p,'frontiers/'+str(p.relative_to(HERE/'static_subtraction_frontiers')))
  else:large.append(dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p),retained_on_remote=True))
for p in (HERE/'link_static_subtraction_models').rglob('validation.json'):copy(p,'link_validation/'+str(p.relative_to(HERE/'link_static_subtraction_models')))
for pattern in ['*static*.log']:
 for p in HERE.glob(pattern):copy(p,'logs/'+p.stem+'.txt')
for name in ['static_subtraction_protocol.json','check_static_subtraction.py','check_balanced_ci.py','check_static_small_molecules.py','qualify_static_model.py','qualify_subtracted_observables.py','run_static_model_validation.py','complete_static_frontier.py','collect_static_candidate.py','test_static_subtraction.py','test_full_ci_ground.py','test_canonical_128.py','test_native_diagonal_cache.py']:copy(HERE/name,'qualified_sources/'+name)
for p in (BASE/'source/packages/matrix-gfn2/src/matrix_gfn2').glob('*.py'):copy(p,'qualified_sources/matrix_gfn2/'+p.name)
(out/'large_artifacts.json').write_text(json.dumps(large,indent=2)+'\n')
files={str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'}
(out/'artifact_manifest.json').write_text(json.dumps(dict(host=socket.getfqdn(),file_count=len(files),source_scope='Source snapshot at export; algorithm generations are distinguished by committed source and per-result eigensolver certificates',sha256=files),indent=2)+'\n');print('ARCHIVED',len(files),flush=True)
