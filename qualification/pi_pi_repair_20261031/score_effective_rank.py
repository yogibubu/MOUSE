"""Score the declared rank proposal using identical completed vertical points.
No electronic jobs. All labels were exposed before this development proposal.
"""
from pathlib import Path
import json,hashlib
import numpy as np
Q=Path(__file__).resolve().parent;root=Q.parents[1];snapshot=Q/'remote_snapshot/pi_pi_repair'
selection_path=snapshot/'effective_rank_selection.json';base_path=root/'qualification/state_following_20261031/remote/completed_14_state_snapshot/energy_scores.json'
selection=json.loads(selection_path.read_text());baseline=json.loads(base_path.read_text())['rows']
attempts={'B016':('Cyclopentadiene','0','1'),'B022':('Furan','0','1'),'B024':('Phthalimide','0_fr_v1','1_gmf_polish_v1')}
rows=[];hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [selection_path,base_path,Q/'effective_rank_protocol.json']}
for rank in selection['rows']:
    b=next(x for x in baseline if x['state']==rank['state'])
    if rank['rank']==1:
        energy=b['oo_ev'];reuse='identical single-CSF functional; locked original result'
    else:
        name,*folders=attempts[rank['state']];parents=[]
        for folder in folders:
            p=snapshot/'optimized'/name/rank['state']/folder/'result.json';r=json.loads(p.read_text());assert r['rank']==rank['rank'] and r['status']=='stationary_candidate';parents.append(r['excitation_ev']);hashes[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
        energy=float(np.mean(parents));reuse='same-rank converged bare-coupling multi-CSF parents; fixed 50/50'
    rows.append(dict(**rank,reference_ev=b['reference_ev'],family=b['family'],baseline_ev=b['oo_ev'],candidate_ev=energy,tda_q_ev=b['tda_selected_ev'],reuse=reuse,error_ev=energy-b['reference_ev']))
metrics={}
for family in ['all','n_pi','pi_pi','CT','Rydberg']:
    subset=[r for r in rows if family=='all' or r['family']==family]
    metrics[family]=dict(n=len(subset),baseline_mae=float(np.mean([abs(r['baseline_ev']-r['reference_ev']) for r in subset])),candidate_mae=float(np.mean([abs(r['error_ev']) for r in subset])),tda_q_mae=float(np.mean([abs(r['tda_q_ev']-r['reference_ev']) for r in subset])))
report=dict(scope='exposed-label development; vertical recombination of identical-rank stationary checkpoints, not independent validation or complete PES integration',protocol=json.loads((Q/'effective_rank_protocol.json').read_text()),input_sha256=hashes,rows=rows,metrics=metrics)
(Q/'effective_rank_development.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(metrics,indent=2))
