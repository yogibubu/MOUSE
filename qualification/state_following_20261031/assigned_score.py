"""Read numerical labels only after immutable reduced predictions are closed."""
from pathlib import Path
import socket,json,hashlib,datetime,csv,math
assert socket.getfqdn()=='oracle.sns.it'
R=Path('/home/enzo/MOUSE-state-following-20261031');P=Path('/home/enzo/MOUSE-publication-validation-20261030');D=Path('/home/enzo/MOUSE-orbital-performance-20261031')
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
lock=read(D/'prediction_lock.json');ledger=read(D/'ledger.json');assert ledger['closed'] and len(ledger['rows'])==14 and lock['protocol_sha256']==sha(D/'protocol.json')
assert all(sha(Path(p))==s for p,s in lock['files_sha256'].items())
scopeledger=read(D/'scope_ledger.json');assert len(scopeledger['rows'])==44 and all(x['status'] in ['matched','deferred_user_scope_reduction'] for x in scopeledger['rows'])
if (D/'energy_scores.json').exists():raise SystemExit(0)
# No code above this point reads the numerical labels.
labels=read(P/'blind/sealed_labels.json')
(D/'labels_opened.json').write_text(json.dumps(dict(opened_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selected_ids=[r['id'] for r in ledger['rows']],prediction_lock_sha256=sha(D/'prediction_lock.json'),scope_note='reduced after baseline convergence outcomes; remaining states deferred; no error-based state selection'),indent=2)+'\n')
rows=[];development={'B001','B011','B015','B016','B022','B039'}
for entry in ledger['rows']:
 sid=entry['id'];name=entry['molecule'];reference=float(labels[sid]['tbe_ev']);assert math.isfinite(reference)
 oo=entry['result'];tda=next(x for x in read(D/'tda'/name/'result.json')['rows'] if x['state']==sid)
 r=dict(state=sid,molecule=name,family=entry['family'],cohort='solver_development' if sid in development else ('solver_repair' if sid in {'B004','B029'} else 'remaining_molecules'),reference_ev=reference,oo_ev=oo['excitation_ev'],oo_error_ev=oo['excitation_ev']-reference,tda_status=tda['status'])
 if tda['status']=='matched':
  r.update(tda_full_ev=tda['parent_energies_ev'][0],tda_selected_ev=tda['parent_energies_ev'][1],tda_parent_mean_ev=tda['mean_parent_tda_ev'])
 rows.append(r)
def stats(data,field):
 errs=[r[field]-r['reference_ev'] for r in data if field in r]
 return dict(qualified=len(errs),requested=len(data),mae_ev=sum(abs(x) for x in errs)/len(errs),rmse_ev=math.sqrt(sum(x*x for x in errs)/len(errs)),signed_mean_ev=sum(errs)/len(errs),maximum_absolute_ev=max(map(abs,errs))) if errs else dict(qualified=0,requested=len(data))
groups={'all_selected':rows,'solver_development':[x for x in rows if x['cohort']=='solver_development'],'remaining_molecules':[x for x in rows if x['cohort']=='remaining_molecules']}
groups['solver_repair']=[x for x in rows if x['cohort']=='solver_repair']
for family in {r['family'] for r in rows}:groups[family]=[r for r in rows if r['family']==family]
methods=['oo_ev','tda_full_ev','tda_selected_ev','tda_parent_mean_ev']
report=dict(host=socket.getfqdn(),rows=rows,statistics={g:{m:stats(data,m) for m in methods} for g,data in groups.items()},paired_comparison={g:{m:stats([r for r in data if r['tda_status']=='matched'],m) for m in methods} for g,data in groups.items()},weights=[.5,.5],lambda_exchange=.9,energy_offset_ev=0.,post_score_fitting=False,scope='14 selected states on 10 molecules; scope reduced after seeing baseline convergence, all prior failures retained separately',tda_mean_is_arithmetic_parent_average=True)
(D/'energy_scores.json').write_text(json.dumps(report,indent=2)+'\n')
fields=sorted(set().union(*(r.keys() for r in rows)))
with (D/'energy_scores.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
print(json.dumps(report['statistics'],indent=2))
