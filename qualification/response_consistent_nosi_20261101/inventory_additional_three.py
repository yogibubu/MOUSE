"""Read-only host, input, method and checkpoint audit for the three-case extension."""
from pathlib import Path
import json,hashlib,socket,subprocess
assert socket.getfqdn()=='oracle.sns.it'
ROOT=Path('/home/enzo/MOUSE-response-consistent-nosi-20261101');CAM=Path('/home/enzo/MOUSE-cam-response-20261101');NATIVE=Path('/home/enzo/MOUSE-native-reference-20261101');BASE=Path('/home/enzo/MOUSE-frontier-cisd-20261101')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(4194304),b''):h.update(block)
 return h.hexdigest()
expected=json.loads((ROOT/'additional_three_input_hashes.json').read_text())['input_files']
for path,digest in expected.items():assert sha(ROOT/'data/additional_three'/path)==digest,path
sources=[ROOT/name for name in ['additional_three.py','selected_extension.py','nosi_spectrum.py','export_selected_model.py','validate_portable.py','audit_additional_spin.py']]
sources +=[CAM/name for name in ['anchor.py','calibrate.py','full_reference.py','native_batch.py','predict.py','heldout_predict.py','nosi_coherent.py','nosi_projected.py']]+[NATIVE/'run.py',BASE/'run.py',BASE/'tda_control.py']
files={}
for prefix in [CAM/'additional_three',NATIVE/'additional_three',BASE/'additional_three',ROOT/'selected_cid/additional_three',ROOT/'prepared_models/selected_cid/additional_three']:
 for path in sorted(prefix.rglob('*')):
  if not path.is_file() or 'grid' in path.parts or path.suffix not in {'.json','.npz','.chk'}:continue
  files[str(path)]=dict(bytes=path.stat().st_size,sha256=sha(path))
(ROOT/'additional_three_remote_inventory.json').write_text(json.dumps(dict(host=socket.getfqdn(),cohort_freeze_commit='f8a38e3',candidate_freeze_commit='467a9e5',unchanged_inputs_verified=len(expected),source_hashes={str(p):sha(p) for p in sources},remote_checkpoints=files,grid_cache_excluded=True),indent=2)+'\n')
print('Verified raw inputs and inventoried source/checkpoints.',flush=True)
