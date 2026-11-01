"""Standalone scientific figure from bundled completed predictions."""
from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
P=Path(__file__).resolve().parent
rows=list(csv.DictReader((P/'data/selected_cid/predictions.csv').open()))
summary=json.loads((P/'data/selected_cid/summary.json').read_text());nd=summary['development']['states'];npredict=summary['prospective']['states']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,(ax,bx)=plt.subplots(1,2,figsize=(11.3,max(5.5,.29*len(rows)+1.15)),gridspec_kw={'width_ratios':[1,1.35]})
r=np.array([float(x['reference_ev']) for x in rows]);o=np.array([float(x['OO_ev']) for x in rows]);t=np.array([float(x['TDA_ev']) for x in rows])
ax.plot([1.5,9],[1.5,9],color='#94a3b8',lw=1)
ax.scatter(r,t,c='#a1a8b2',marker='x',s=35,label='TDA: same native orbitals/kernel')
for i,row in enumerate(rows):
 prospective=row['cohort'] in {'cid_prospective','additional_three','last_two'}
 ax.scatter(r[i],o[i],marker='s' if prospective else 'o',s=40,facecolor='#147d92' if prospective else '#254c7b',edgecolor='white',lw=.5,label='Selected OO-CID' if i==0 else None)
ax.set(xlabel='Reference excitation / eV',ylabel='Calculated excitation / eV',xlim=(1.5,max(6.5,float(r.max())+.2)),ylim=(1.5,max(9.2,float(t.max())+.3)))
ax.legend(frameon=False,fontsize=8,loc='upper left');ax.set_title('(a) Matched native-orbital comparison',loc='left',fontweight='bold',fontsize=10)
bx.axvspan(-.5,.5,color='#eef3f7');bx.axvline(0,c='#8998aa',lw=.8)
for i,row in enumerate(rows):
 prospective=row['cohort'] in {'cid_prospective','additional_three','last_two'};color='#147d92' if prospective else '#254c7b'
 bx.plot([0,o[i]-r[i]],[i,i],c=color,lw=2)
 bx.scatter(o[i]-r[i],i,c=color,marker='s' if prospective else 'o',s=32,zorder=3)
bx.set(yticks=np.arange(len(rows)),yticklabels=[x['id']+'  '+x['molecule'].replace('_Intermediate','').replace('_',' ') for x in rows],xlabel='Selected OO-CID signed error / eV',xlim=(-max(.55,float(abs(o-r).max())+.05),max(.55,float(abs(o-r).max())+.05)));bx.invert_yaxis();bx.axhline(nd-.5,c='#64748b',ls='--',lw=.8)
bx.set_title(f'(b) {nd} development + {npredict} post-freeze states',loc='left',fontweight='bold',fontsize=10)
fig.text(.03,.025,f"Fixed 50/50 weights; lambda = 0.9; no fitted energy offset. MAE {summary['all']['OO_MAE_ev']:.3f} eV, maximum {summary['all']['OO_max_ev']:.3f} eV. Squares: post-freeze predictions.",fontsize=8,color='#334155')
fig.tight_layout(rect=(0,.05,1,1),w_pad=2)
for ext in ['pdf','png']:fig.savefig(P/'figures'/('selected_cid_validation.'+ext),dpi=220,bbox_inches='tight')
lines=[r'\begin{tabular}{llrrr}\toprule',r'ID & Molecule & Reference & OO-CID & TDA \\ \midrule']
for x in rows:lines.append(f"{x['id']} & {x['molecule'].replace('_',' ')} & {float(x['reference_ev']):.3f} & {float(x['OO_ev']):.3f} & {float(x['TDA_ev']):.3f}" + r' \\')
lines.append(r'\bottomrule\end{tabular}');(P/'tables/selected_cid_validation.tex').write_text('\n'.join(lines)+'\n')
