"""Score only locked exposed-development results; never used by model builders."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'final_validation_20261031/remote_final'
PREVIOUS=ROOT.parent/'transferability_20261101/remote/stationary_analytic/orthogonal'
rows=[]
for cohort,name in [('independent','Criegee_Intermediate'),('independent_challenge','Furofuran')]:
    refs={r['id']:r for r in json.loads((OLD/cohort/'score.json').read_text())['rows']}
    for path in sorted((ROOT/'remote'/cohort/name/'common_states').glob('*/result.json')):
        d=json.loads(path.read_text());assert d['qualified'];ref=refs[d['id']]
        previous=json.loads((PREVIOUS/cohort/name/d['id']/'result.json').read_text())
        rows.append(dict(id=d['id'],molecule=name,family=d['family'],reference_ev=ref['reference_ev'],
            original_ev=ref['excitation_ev'],orthogonal_previous_ev=previous['excitation_ev'],
            stationary_common_ev=d['excitation_ev'],stationary_common_error_ev=d['excitation_ev']-ref['reference_ev'],
            tda_ev=d['tda']['excitation_ev'],tda_qualified=d['tda']['qualified'],
            kkt_max=d['kkt_max'],fidelity=d['fidelity'],ground_overlap_squared=d['ground_overlap_squared'],
            result_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
assert len(rows)==4
methods={}
for key in ['original_ev','orthogonal_previous_ev','stationary_common_ev','tda_ev']:
    errors=[r[key]-r['reference_ev'] for r in rows]
    methods[key]=dict(n=4,mae_ev=sum(abs(x) for x in errors)/4,rmse_ev=(sum(x*x for x in errors)/4)**.5,maxabs_ev=max(abs(x) for x in errors))
report=dict(scope='four already exposed states in two development molecules; no blind or universal claim',rows=rows,
    metrics=methods,weights=[.5,.5],lambda_exchange=.9,offset_ev=0,
    tda_comparison='same new corrected model and own stationary orbitals, derived from the embedded-DFT anchor; not native unmodified GFN2-orbital TDA',
    conclusion='Stationary-anchor and common-functional numerical coherence established; accuracy does not improve over the preceding ground-orthogonal model; do not promote as transferability correction',
    limitations=['rank one', 'no new ground-orthogonality nuclear response', 'compact representation limitations unchanged',
                 'common functional differs from averaging separately optimized parent energies'])
(ROOT/'score.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(methods,indent=2))
