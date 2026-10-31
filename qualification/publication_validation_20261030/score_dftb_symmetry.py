"""Score frozen symmetry assignments on the unchanged original 19-state list."""
from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parents[2]
Q=Path(__file__).resolve().parent/'remote_dftb_symmetry_20261031'
csvpath=ROOT/'manuscripts/mouse_orbital_optimized_dft/data/vertical_energies.csv'
td=json.loads((Q/'report.json').read_text());delta=json.loads((Q/'delta_report.json').read_text())
targets=list(csv.DictReader(csvpath.open()))
tl={(r['molecule'],r.get('state')):r for r in td['rows'] if r['status']=='symmetry_ordinal_assigned'}
dl={(r['molecule'],r.get('state')):r for r in delta['rows']}
rows=[]
for t in targets:
    key=(t['molecule'],t['state']);a=tl.get(key);d=dl[key]
    rows.append(dict(molecule=key[0],state=key[1],cohort=t['cohort'],tbe_ev=float(t['tbe_ev']),
                     oo_dft_ev=float(t['oo_dft_ev']),td_dftb_ev=a['assigned']['energy_ev'] if a else None,
                     delta_dftb_ev=d.get('assigned',{}).get('gap_ev'),td_assignment='symmetry/ordinal only' if a else 'unassigned',
                     delta_assignment=d['status']))
def score(method,cohort=None):
    subset=[r for r in rows if r[method+'_ev'] is not None and (cohort is None or r['cohort']==cohort)]
    return dict(count=len(subset),mae_ev=sum(abs(r[method+'_ev']-r['tbe_ev']) for r in subset)/len(subset),
                oo_same_subset_mae_ev=sum(abs(r['oo_dft_ev']-r['tbe_ev']) for r in subset)/len(subset))
stats={}
for method in ['td_dftb','delta_dftb']:
    stats[method]=score(method)
    stats[method].update({cohort:score(method,cohort) for cohort in ['regression','prospective']})
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out=dict(scope='original curated cohort; TD/Delta-DFTB2 mio-1-1; same archived geometries; symmetry/ordinal assignments without cross-model transition-overlap qualification; all incomplete/inapplicable states retained',
         rows=rows,statistics=stats,source_sha256=sha(Path(__file__)),
         input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [csvpath,Q/'report.json',Q/'delta_report.json']},
         additional_reference_targets_not_in_original_cohort=[dict(molecule=k[0],state=k[1],reason='not in the unchanged original 19-state cohort') for k in tl if k not in {(r['molecule'],r['state']) for r in rows}])
(Q/'comparison.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(stats,indent=2))
