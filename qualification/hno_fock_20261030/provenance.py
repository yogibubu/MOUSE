import json,hashlib,socket
from pathlib import Path
BASE=Path('/home/enzo/MOUSE-hno-fock-20261030')
assert socket.getfqdn()=='oracle.sns.it'
rows=[]
for path in sorted(BASE.rglob('*')):
 if not path.is_file() or path.name.startswith('._') or '__pycache__' in path.parts or path.suffix in {'.pyc','.so'} or path.name in {'provenance.json','transfer_paths.txt'}:continue
 if 'source/packages' in str(path) and path.name not in {'calculator.py','response_reference.py','optimizer.py'}:continue
 rows.append(dict(path=str(path.relative_to(BASE)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size))
record=dict(host=socket.getfqdn(),artifacts=rows,heavy_local_calculations=False,
 source_note='Final production calculator/reference/optimizer sources are retained. Intermediate algorithms are documented separately; the damped and inner-Newton experiments are archived but not included in final production. No single final source snapshot is claimed to be the exact source of every earlier experiment.',
 parent_checkpoint_root='/home/enzo/MOUSE-roks-following-20261030',all_jobs_complete=True,certified_minima=0)
(BASE/'provenance.json').write_text(json.dumps(record,indent=2)+'\n');print(len(rows),'artifacts')
