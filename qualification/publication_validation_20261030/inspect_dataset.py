from pathlib import Path
import socket,json,collections
assert socket.getfqdn()=='oracle.sns.it'
p=Path(__file__).resolve().parent/'reference/QUESTDB'
fs=list((p/'data/json').rglob('*.json'));ty=collections.Counter();fields=collections.Counter();groups=collections.Counter()
for f in fs:
 d=json.loads(f.read_text());rs=d if isinstance(d,list) else [d]
 for r in rs:
  if isinstance(r,dict):ty.update([str(r.get('Type'))]);fields.update(r.keys());groups.update([str(f.parent.name)])
print(json.dumps(dict(files=len(fs),types=dict(ty),field_names=list(fields),groups=dict(groups),molecule_filenames=sorted(set(f.stem for f in fs))),indent=2))
