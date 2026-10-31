"""Read-only campaign ledger; do not open sealed labels or recompute electronics."""
from pathlib import Path
import json,socket,hashlib,datetime
assert socket.getfqdn()=='oracle.sns.it'
R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
anchors=[r for p in sorted((R/'anchor_energy').glob('*/*/result.json')) if 'change_ev' in (r:=read(p))]
manifest=read(R/'independent/manifest.json');predictions=[]
for row in manifest['states']:
 p=R/'independent/predictions'/row['molecule']/row['id']/'result.json'
 predictions.append(read(p) if p.exists() else dict(id=row['id'],molecule=row['molecule'],family=row['family'],status='pending'))
geometry=[read(p) for p in sorted((R/'geometry').glob('*/result.json'))]
cost=[read(p) for p in sorted((R/'cost').glob('*/*/*/result.json'))]
report=dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=socket.getfqdn(),protocol_sha256=sha(R/'protocol.json'),anchor_energy=dict(complete=len(anchors)==6,rows=anchors,maximum_absolute_change_ev=max([abs(r['change_ev']) for r in anchors],default=None)),independent=dict(manifest_sha256=sha(R/'independent/manifest.json'),labels_read=False,all_attempts_closed=all(r['status']!='pending' for r in predictions),rows=predictions),structure_frequency=dict(completed=len(geometry),rows=geometry,physical_reference_comparison='pending until comparison certificate exists'),cost=dict(completed=sum(r['status']=='complete' for r in cost),attempts_closed=len(cost),expected=72,rows=cost,full_preparation_cost_coverage='requires separate timing-source audit; not inferred from property-only measurements'))
(R/'status.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(anchors=len(anchors),independent_closed=sum(r['status']!='pending' for r in predictions),geometries=len(geometry),cost_runs=len(cost),labels_read=False)))
