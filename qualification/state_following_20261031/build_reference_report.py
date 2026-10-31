"""Lightweight figures and checked report from completed Oracle JSON evidence."""
from pathlib import Path
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;D=P/'remote/candidate_reference'
def read(p):return json.loads(p.read_text())
controls=read(D/'ritz_projection_probe/validation_report.json');selection=read(D/'consistent_reference/report.json')
assert controls['status']=='passed' and len(controls['cases'])==3
source_hashes=read(D/'ritz_projection_probe/controls_protocol.json')['source_sha256']
for name,digest in source_hashes.items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest
assert hashlib.sha256((P/'select_consistent_reference.py').read_bytes()).hexdigest()==selection['protocol']['source_sha256']
assert len(selection['rows'])==6 and all(x['status']=='matched' for x in selection['rows'])
for row in selection['rows']:
 assert (row['original_canonical_gap_hartree']>0)==(row['selected_profile']=='original_qualified_spectral')
assert sum(x['selected_profile']=='complete_projected_ground_fock' for x in selection['rows'])==1
old=read(D/'spectral_ordering_probe/ground_chart/chart_comparison.json');new=read(D/'ritz_projection_probe/ground_chart/chart_comparison.json')
assert new['ground']['index']==0 and new['ground']['minimum_curvature']>0
assert new['charts'][1]['projection_norm']>=.7
report=dict(host='oracle.sns.it',local_sources_match_tested_remote=True,selected_reference_numerical_cases=selection['rows'],full_projected_fock_controls=controls['cases'],weights=[.5,.5],lambda_exchange=.9,energy_offset_ev=0.,bimane_ground_transition_projection=dict(partial_levels=old['charts'][1]['projection_norm'],complete_projection=new['charts'][1]['projection_norm']),accuracy_scoring_performed=False,limitations=['exploratory model amendment after seeing baseline convergence/representation outcomes','no numerical excitation reference labels used','Bimane and aza-naphthalene have no competitor root in their targeted subsets','numerical qualification does not demonstrate smaller energy error','new one-body profile nuclear derivatives have not yet been validated','reference selection is verified on six completed cases, not enabled as a global production default'])
(P/'REFERENCE_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
navy='#183F54';teal='#148681';orange='#D4874D';gray='#7A8791'
fig,(a,b)=plt.subplots(1,2,figsize=(9.3,3.9));fig.subplots_adjust(left=.08,right=.98,top=.83,bottom=.25,wspace=.35)
for root,col,label in [('spectral_ordering_probe',orange,'Partial level transfer'),('ritz_projection_probe',teal,'Complete Fock projection')]:
 h=read(D/root/'results/Bimane_Syn/B015/accepted.json')['history'];h=[x for x in h if x['accepted']]
 a.plot([x['to_fraction'] for x in h],[x['guard']['sector_norm'] for x in h],'o-',color=col,lw=1.8,ms=4,label=label)
a.axhline(.7,color=gray,ls='--',lw=1);a.set(xlabel='One-body interpolation fraction',ylabel='Minimum occupied/virtual sector norm',ylim=(.65,1.025),xlim=(.17,1.03));a.set_title('a  Bimane: preserve the assigned orbital sectors',loc='left',fontsize=10,color=navy);a.legend(frameon=False,fontsize=8,loc='lower left',bbox_to_anchor=(0.,.22))
vals=[old['charts'][0]['projection_norm'],old['charts'][1]['projection_norm'],new['charts'][1]['projection_norm']]
b.bar([0,1,2],vals,color=[navy,orange,teal],width=.63)
for x,v in enumerate(vals):b.text(x,v+.025,f'{v:.3f}',ha='center',fontsize=9,color=navy)
b.axhline(.7,color=gray,ls='--',lw=1);b.set_xticks([0,1,2],['Native\nreference','Relaxed S0\npartial levels','Relaxed S0\ncomplete Fock']);b.set(ylim=(0,1.06),ylabel='Projected DFT transition norm');b.set_title('b  A consistent relaxed ground reference',loc='left',fontsize=10,color=navy)
fig.text(.5,.035,'Distinct one-body operators; identical charge/spin response, λ = 0.9 and 50/50 weights. No energy offset.',ha='center',fontsize=8,color=gray)
for ext in ['pdf','svg','png']:fig.savefig(P/f'consistent_reference_repair.{ext}',dpi=230,bbox_inches='tight',pad_inches=.15)
plt.close(fig)
print(json.dumps(dict(selected_cases=6,projected_fock_controls=3,bimane_ev=next(x['excitation_ev'] for x in selection['rows'] if x['molecule']=='Bimane_Syn'),accuracy_scored=False),indent=2))
