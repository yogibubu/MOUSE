"""Export public state order/symmetry only; never open segregated scoring labels."""
from pathlib import Path
import socket,json,hashlib
assert socket.getfqdn()=='oracle.sns.it'
P=Path(__file__).resolve().parent;dest=P/'blind/assignment_catalog.json'
if dest.exists():raise SystemExit('Assignment catalog already frozen')
manifest=json.loads((P/'blind/manifest.json').read_text());out=[]
for s in manifest['states']:
 source=P/'reference/QUESTDB'/s['source_metadata']
 # Source metadata is read in this separate export step, never by predictors.
 # Numerical reference fields are discarded and never used for root ordering.
 rows=[{'state':r.get('State'),'spin':r.get('Spin'),'type':r.get('Type'),'vr':r.get('V/R')} for r in json.loads(source.read_text())]
 i=s['metadata_index'];assert rows[i]['state']==s['state_label'] and rows[i]['spin']==1
 ordinal=sum(r['state']==s['state_label'] and r['spin']==1 for r in rows[:i+1])
 out.append(dict(id=s['id'],molecule=s['molecule'],state_label=s['state_label'],family=s['family'],ordinal_same_singlet_symmetry=ordinal,public_metadata_index=i))
value=dict(rule='Within each singlet symmetry, use published metadata row order; never TBE values or closest-energy assignment. Degenerate/uncorrelated point groups remain unassigned until separately audited.',manifest_sha256=hashlib.sha256((P/'blind/manifest.json').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),fields_read=['State','Spin','Type','V/R'],numerical_reference_fields_used=False,states=out)
with dest.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
print('Frozen public assignment catalog:',len(out),'states; numerical reference values not exported')
