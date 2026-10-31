"""Lightweight status collection; never open blind scoring labels."""
from pathlib import Path
import socket,json,datetime
assert socket.getfqdn()=='oracle.sns.it'
P=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text()) if p.exists() else None
manifest=read(P/'blind/manifest.json');reference=[]
for name in sorted(set(r['molecule'] for r in manifest['states'])):
 d=P/'blind/references'/name;r=read(d/'result.json');f=read(d/'failure.json')
 reference.append(dict(molecule=name,status=r['status'] if r else f['status'] if f else 'pending',roots=r.get('roots') if r else None,covers_20_ev=r.get('covers_20_ev') if r else None))
dftb=[]
for d in sorted((P/'external/dftbplus24/attempt_03').glob('*/*/result.json')):
 r=read(d);recovery=P/'external/dftbplus24/attempt_04'/r['molecule']/r['method']/'result.json'
 if recovery.exists():r=read(recovery)
 dftb.append(dict(molecule=r['molecule'],method=r['method'],numerical_status=r['status'],state_assignment='pending',qualified_comparison=False))
s=read(P/'direct_stda_comparison.json')
mouse_rows=[]
for f in sorted((P/'blind/mouse').glob('*/result.json')):
 r=read(f);mouse_rows.extend(r.get('rows',[]))
mouse_counts={}
for r in mouse_rows:mouse_counts[r['status']]=mouse_counts.get(r['status'],0)+1
result=dict(pipeline=read(P/'blind/pipeline_progress.json'),mouse_state_outcomes=mouse_counts,as_of_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=socket.getfqdn(),model=dict(weights=[.5,.5],lambda_exchange=.9,offset_ev=0.,unchanged=True),direct_stda=dict(requested=s['requested'],matched=s['matched'],mae_ev=s['mae_ev'],scope='original curated cohort, not new blind set'),dftb=dftb,blind=dict(states=len(manifest['states']),molecules=len(reference),counts=manifest['counts'],reference_progress=read(P/'blind/references/progress.json'),references=reference,mouse_progress=read(P/'blind/mouse/progress.json'),mouse_preflight=read(P/'blind/mouse/preflight.json'),mouse_protocol_frozen=(P/'blind/mouse/protocol.json').exists(),scoring_labels_opened=False),completed_prechecks=['three-molecule CCSDR(3) internal coordinates','vertical-anchor algebraic weights only','HNO harmonic frequencies in provisional frequency order'],pending=['qualified TD-DFTB and DeltaDFTB state assignment','blind MOUSE prediction and all-state ledger before scoring','calibrated TDA-GFN2','full lambda and anchor sensitivity','additional high-level frequencies and mode assignment','complete cold/warm energy/gradient/Hessian/process-memory cost scaling'])
(P/'STATUS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(host=result['host'],reference_ready=sum(r['status']=='reference_ready' for r in reference),reference_partial=sum(r['status']=='reference_partial' for r in reference),reference_failures=sum(r['status'] not in ['reference_ready','reference_partial','pending'] for r in reference),total_molecules=len(reference),current_reference=result['blind']['reference_progress'],td_dftb_completed=sum(r['method']=='td' and r['numerical_status']=='completed_unassigned' for r in dftb),delta_dftb_completed=sum(r['method']=='delta' and r['numerical_status']=='completed_unassigned' for r in dftb),preflight=result['blind']['mouse_preflight'],pipeline=result['pipeline'],mouse_state_outcomes=mouse_counts),indent=2))
