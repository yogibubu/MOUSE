"""Static scientific figure from archived exposed-label development results."""
from pathlib import Path
import json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;dest=ROOT/'data/reduced_extension/effective_rank_development.json';source=ROOT.parents[1]/'qualification/pi_pi_repair_20261031/effective_rank_development.json'
if source.exists():shutil.copy2(source,dest)
a=json.loads(dest.read_text());order=['n_pi','pi_pi','CT','Rydberg'];rows=sorted(a['rows'],key=lambda r:(order.index(r['family']),r['state']))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,(ax,bx)=plt.subplots(1,2,figsize=(10.5,6.2),gridspec_kw={'width_ratios':[1.7,1]},layout='constrained');y=np.arange(len(rows))
for field,color,marker,offset,label in [('baseline_ev','#767676','o',-.19,'Original OO 50/50'),('candidate_ev','#1670A5','o',0,'Effective-rank trial'),('tda_q_ev','#D07722','x',.19,'Matched TDA-Q')]:
    err=np.array([r[field]-r['reference_ev'] for r in rows]);ax.scatter(err,y+offset,c=color,marker=marker,s=28,label=label,zorder=3)
ax.axvline(0,c='#333333',lw=.8);ax.set_yticks(y,[r['state']+'  '+r['molecule'].replace('_',' ') for r in rows]);ax.invert_yaxis();ax.set_xlabel('Signed excitation-energy error / eV');ax.set_title('(a) Every state in the 14-state extension',loc='left',weight='bold');ax.grid(axis='x',alpha=.18)
for boundary in [4.5,9.5,12.5]:ax.axhline(boundary,c='#cccccc',lw=.6)
ax.legend(loc='lower right',frameon=True,fontsize=8)
labels=['All (14)',r'$n\to\pi^*$ (5)',r'$\pi\to\pi^*$ (5)','CT (3)','Rydberg (1)'];keys=['all']+order;x=np.arange(5);width=.24
for field,color,offset,label in [('baseline_mae','#767676',-width,'Original'),('candidate_mae','#1670A5',0,'Trial'),('tda_q_mae','#D07722',width,'TDA-Q')]:
    bx.bar(x+offset,[a['metrics'][k][field] for k in keys],width,color=color,label=label)
bx.set_xticks(x,labels,rotation=40,ha='right');bx.set_ylabel('Mean absolute error / eV');bx.set_title('(b) Family-resolved errors',loc='left',weight='bold');bx.grid(axis='y',alpha=.18);bx.set_axisbelow(True)
fig.savefig(ROOT/'figures/multicsf_development.pdf');fig.savefig(ROOT/'figures/multicsf_development.png',dpi=180)
