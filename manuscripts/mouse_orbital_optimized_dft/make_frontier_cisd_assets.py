"""Reproduce the exposed/prospective pilot figure from immutable scored values."""
from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
HERE=Path(__file__).resolve().parent;SOURCE=HERE/'data/frontier_cisd'
rows=list(csv.DictReader((SOURCE/'predictions.csv').open()));summary=json.loads((SOURCE/'summary.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,(ax,bx)=plt.subplots(1,2,figsize=(10.8,4.5),gridspec_kw={'width_ratios':[1,1.35]})
r=np.array([float(x['reference_ev']) for x in rows]);oo=np.array([float(x['OO_ev']) for x in rows]);tda=np.array([float(x['TDA_ev']) for x in rows]);y=np.arange(len(rows))
ax.plot([2,8.5],[2,8.5],color='#667085',lw=1,zorder=0)
ax.scatter(r,tda,c='#a9b0bb',marker='x',s=37,label='TDA, same prepared kernel')
for i,row in enumerate(rows):
 prospective=row['cohort']=='prospective_cisd'
 ax.scatter(r[i],oo[i],marker='s' if prospective else 'o',s=38,facecolor='#215c96',edgecolor='#102a43',lw=.7,label='OO + frontier CISD' if i==0 else None)
ax.set(xlabel='Reference excitation energy / eV',ylabel='Calculated excitation energy / eV',xlim=(2,6.4),ylim=(2,8.6));ax.legend(frameon=False,loc='upper left',fontsize=8)
ax.set_title('(a) Identical native GFN2 spatial orbitals',loc='left',fontweight='bold',fontsize=10)
colors=['#bd642d' if x['id'] in ['I001','N001','N002','N003','V001','V002'] else '#215c96' for x in rows]
bx.axvspan(-.5,.5,color='#eaf0f5',zorder=0);bx.axvline(0,c='#667085',lw=.8);bx.axvline(-.5,c='#9aa5b1',ls=':',lw=.9);bx.axvline(.5,c='#9aa5b1',ls=':',lw=.9)
for i,row in enumerate(rows):
 bx.plot([0,oo[i]-r[i]],[i,i],c=colors[i],lw=2)
 bx.scatter(oo[i]-r[i],i,c=colors[i],marker='s' if row['cohort']=='prospective_cisd' else 'o',s=34,zorder=3)
bx.set(yticks=y,yticklabels=[f"{x['id']}  {x['molecule'].replace('_Intermediate','').replace('_',' ')}" for x in rows],xlabel='OO + frontier CISD signed error / eV',xlim=(-.7,.7));bx.invert_yaxis();bx.axhline(8.5,c='#667085',ls='--',lw=.8)
bx.set_title('(b) Nine development + two prospective states',loc='left',fontweight='bold',fontsize=10)
fig.text(.02,.02,'Pilot study: MAE 0.217 eV; maximum error 0.552 eV. Squares: prospective predictions on historically exposed molecules.',fontsize=8,color='#344054')
fig.tight_layout(rect=(0,.045,1,1),w_pad=2)
for ext in ['pdf','png']:fig.savefig(HERE/'figures'/('frontier_cisd_pilot.'+ext),dpi=220,bbox_inches='tight')
lines=['\\begin{tabular}{llrrr}\\hline','ID & Molecule & Reference & OO+CISD & TDA \\\\ \\hline']
for x in rows:lines.append(f"{x['id']} & {x['molecule'].replace('_',' ')} & {float(x['reference_ev']):.3f} & {float(x['OO_ev']):.3f} & {float(x['TDA_ev']):.3f} \\\\")
lines+=['\\hline\\end{tabular}'];(HERE/'tables/frontier_cisd_pilot.tex').write_text('\n'.join(lines)+'\n')
