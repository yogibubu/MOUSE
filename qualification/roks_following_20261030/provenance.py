"""Hash final molecular artifacts and the actual source snapshots retained."""
import socket,json,hashlib,sys
from pathlib import Path
BASE=Path('/home/enzo/MOUSE-roks-following-20261030')
assert socket.getfqdn()=='oracle.sns.it'
rows=[]
for path in sorted(BASE.rglob('*')):
 if path.name.startswith('._') or not path.is_file() or '__pycache__' in path.parts or path.suffix in {'.pyc','.so'} or path.name=='provenance.json':continue
 # Keep all calculation data and drivers. Include source files used by this
 # implementation; other package source is in the preserved parent snapshot.
 if 'source/packages' in str(path):
  if path.name not in {'roks_tracking.py','response_reference.py','composite_roks.py','composite_roks_component.py','scan.py','optimizer.py','qmqm_component.py'}:continue
 rows.append(dict(path=str(path.relative_to(BASE)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size))
record=dict(host=socket.getfqdn(),artifacts=rows,source_note='Initial primary continuations used the adapter preserved as source/qualification/composite_roks_component_initial.py and parent response_reference/qmqm code. Later additions are explicit opt-in; original profiles remain unchanged. The final files alone are not asserted to be the exact original source bytes of every earlier process.',
 parent_source='/home/enzo/MOUSE-roks-50-50-optimization-20261030/source/packages',heavy_local_calculations=False)
(BASE/'provenance.json').write_text(json.dumps(record,indent=2)+'\n')
print(len(rows),'artifacts')
