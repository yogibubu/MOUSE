"""Lightweight scoring of exposed development cohorts; no electronic jobs."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'final_validation_20261031/remote_final'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
reports={}
for cohort in ['independent','independent_challenge']:
    prior=json.loads((OLD/cohort/'score.json').read_text());rows=[]
    for old in prior['rows']:
        row={k:old[k] for k in ['id','molecule','family','status']}
        if 'reference_ev' in old:row['reference_ev']=old['reference_ev']
        p=ROOT/'remote/stationary_analytic/orthogonal'/cohort/old['molecule']/old['id']/'result.json'
        if p.exists():
            result=json.loads(p.read_text());assert result['qualified']
            row.update(status='matched_development_candidate',baseline_ev=old['excitation_ev'],
                candidate_ev=result['excitation_ev'],baseline_error_ev=old['error_ev'],
                candidate_error_ev=result['excitation_ev']-old['reference_ev'],result_sha256=sha(p),
                maximum_kkt_residual=max(x['kkt_max'] for x in result['parents']),
                maximum_ground_overlap_squared=max(x['ground_overlap_squared'] for x in result['parents']),
                minimum_state_fidelity=min(x['fidelity'] for x in result['parents']))
        rows.append(row)
    matched=[r for r in rows if 'candidate_ev' in r]
    reports[cohort]=dict(n_prespecified=len(rows),n_matched=len(matched),rows=rows,
        baseline_mae_ev=sum(abs(r['baseline_error_ev']) for r in matched)/len(matched),
        candidate_mae_ev=sum(abs(r['candidate_error_ev']) for r in matched)/len(matched))
result=dict(schema='mouse.transferability.development.score.v1',host_for_electronic_work='oracle.sns.it',
    weights=[.5,.5],lambda_exchange=.9,offset_ev=0,cohorts=reports,
    assessment='Numerical state-collapse safeguard demonstrated; accuracy and representation transferability remain unresolved',
    limitations=['All target labels were exposed before this development campaign',
        'Separately optimized parents; not yet the production relative-parent manifold',
        'New ground-orthogonality nuclear response not implemented',
        'Compact-basis failures and higher-rank cases unchanged; none silently removed'])
(ROOT/'score.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:{a:v[a] for a in ['n_prespecified','n_matched','baseline_mae_ev','candidate_mae_ev']} for k,v in reports.items()},indent=2))
