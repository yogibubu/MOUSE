"""Lightweight report and figures from completed remote results only."""
from pathlib import Path
import json,hashlib,re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;D=P/'remote';repo=P.parents[1]
validation=json.loads((D/'validation_progress.json').read_text());assert validation['status']=='passed'
audit=json.loads((D/'recovered_assignment_audit.json').read_text());assignment={r['state']:r for r in audit['rows']}
rows=[];controls=[]
for file in sorted((D/'recovery_rank_one').glob('*/*/result.json')):
 r=json.loads(file.read_text());a=assignment[r['state']];assert r['status']=='matched' and a['uniquely_assigned']
 previous=r.get('baseline_result');old=next((x for x in previous.get('rows',[]) if x['id']==r['state'] and x['status']=='matched'),None) if previous else None
 row=dict(molecule=r['molecule'],state=r['state'],classification='positive_control' if old else 'exploratory_recovery',energy_ev=r['excitation_ev'],fidelity=r['final_guard']['transition'],raw_transition=r['final_guard']['raw_transition'],best_single_pair_overlap=r['final_guard']['rank_one_limit'],joint_residual=r['joint_residual'],joint_index=r['joint_index'],minimum_orbital_curvature_hartree=r['minimum_curvature'],minimum_competitor_margin=min(a['projected_overlap_margin_against_all_converged_same_symmetry_roots']),available_competing_reference_roots=len(a['available_same_symmetry_candidates'])-1,reference_coverage_complete=False,endpoint_energy_check_hartree=r['endpoint_energy_check_hartree'])
 if old:
  row['control_energy_change_ev']=r['excitation_ev']-old['excitation_ev'];controls.append(abs(row['control_energy_change_ev']))
 rows.append(row)
assert len(rows)==5
source={}
for name in ['orbital_saddle.py','roks_orbital_response.py','roks_tracking.py','roks_continuation.py']:
 remote=D/'engine'/name;local=repo/'packages/matrix-gfn2/src/matrix_gfn2'/name
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 source[name]=dict(local_sha256=sha(local),tested_remote_sha256=sha(remote),match=sha(local)==sha(remote))
assert all(r['match'] for r in source.values())
tests=int(re.search(r'(\d+) passed', (D/'regression_tests.log').read_text())[1])
libs=[json.loads(f.read_text()) for f in (D/'library_validation').glob('*/*/result.json')];assert len(libs)==2 and all(r['passed'] for r in libs)
report=dict(host='oracle.sns.it',date='2026-10-31',scope='state assignment and solver reliability; not an excitation-energy accuracy result',recovered_states=sum(r['classification']=='exploratory_recovery' for r in rows),positive_controls=len(controls),maximum_control_energy_change_ev=max(controls),tests_passed=tests,library_checkpoint_tests=libs,rows=rows,source_verification=source,weights=[.5,.5],lambda_exchange=.9,energy_offset_ev=0.,original_blind_baseline_preserved=True,reference_labels_unopened=True,limitations=['only five molecular states tested with this recovery','no competitor roots available for aza-naphthalene in its targeted reference subset','comparison with available roots does not prove global uniqueness','seven compact-basis projection failures are not repaired by a better solver','full 44-state validation and accuracy scoring remain unfinished'])
(P/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
# Presentation of recorded diagnostics only; no molecular calculation is run.
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
navy='#173D52';teal='#168A8A';orange='#D98745';purple='#8877AE';gray='#75838E'
fig,(a,b)=plt.subplots(1,2,figsize=(9.2,3.7));fig.subplots_adjust(left=.08,right=.98,bottom=.25,top=.83,wspace=.36)
def hist(name,sid):return [r for r in json.loads((D/'recovery_rank_one'/name/sid/'history.json').read_text()) if r['accepted']]
h=hist('Cyclopentadiene','B016');x=[r['to_fraction'] for r in h]
for key,col,label,style in [('raw_transition',orange,'Raw transition overlap','o-'),('transition',teal,'Single-pair fidelity','s-'),('local_csf',navy,'Local CSF continuity','D-')]:a.plot(x,[r['guard'][key] for r in h],style,color=col,lw=1.6,ms=4,label=label)
a.axhline(.7,color=gray,lw=1,ls='--');a.text(.21,.704,'Original raw-overlap gate',fontsize=7,color=gray)
a.set(xlabel='One-body interpolation fraction t',ylabel='Overlap / fidelity',ylim=(.65,1.025),xlim=(.18,1.02));a.set_title('a  Cyclopentadiene: continuous assigned branch',loc='left',fontsize=10,color=navy);a.legend(frameon=False,fontsize=7.5,loc='center left',bbox_to_anchor=(0.,.62))
for name,sid,col,label in [('Cyclopentadiene','B016',teal,'Cyclopentadiene'),('Benzoquinone','B039',orange,'Benzoquinone'),('Aza-naphthalene','B011',purple,'Aza-naphthalene')]:
 h=hist(name,sid);b.plot([r['to_fraction'] for r in h],[r['index'] for r in h],'o--',color=col,lw=1.5,ms=4,label=label)
b.set(xlabel='One-body interpolation fraction t',ylabel='Certified electronic saddle index',ylim=(1.6,5.45),xlim=(.18,1.02));b.set_yticks([2,3,4,5]);b.set_title('b  Follow character while certifying the index',loc='left',fontsize=10,color=navy);b.legend(frameon=False,fontsize=7.5,loc='center left')
fig.text(.5,.035,'t guides initialization; final energy models and 50/50 composite weights are unchanged.',ha='center',fontsize=8,color=gray)
for ext in ['pdf','svg','png']:fig.savefig(P/('state_following_recovery.'+ext),dpi=230,bbox_inches='tight',pad_inches=.13)
plt.close(fig)
print(json.dumps(dict(recovered=report['recovered_states'],controls=report['positive_controls'],tests=tests,max_control_change_ev=max(controls),engine_hashes_match=True),indent=2))
