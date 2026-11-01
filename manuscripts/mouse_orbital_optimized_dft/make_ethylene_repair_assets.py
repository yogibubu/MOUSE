"""Plot the calibrated extension using archived scalar results only."""
from pathlib import Path
import json
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;D=P/'data/closing_validation';s=json.loads((D/'empirical_bath_summary.json').read_text());old=json.loads((P/'data/screened_cid/predictions.json').read_text())+json.loads((D/'prospective_summary.json').read_text())['rows'];before={r['id']:r['error_ev'] for r in old};allrows=sorted(s['rows'],key=lambda r:(r['family']!='pi_pi',r['molecule'],r['id']));rows=[r for r in allrows if r['family']=='pi_pi']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
f,(a,b)=plt.subplots(1,2,figsize=(10,5.8),gridspec_kw={'width_ratios':[1.35,1]});y=np.arange(len(rows));a.axvspan(-.5,.5,color='#eff5f5');a.axvline(0,color='#a5b2bd',lw=.7)
for i,r in enumerate(rows):
 if r['id']=='B002':a.axhspan(i-.44,i+.44,color='#d4ebe7',zorder=0)
 a.plot([before[r['id']],r['error_ev']],[i,i],color='#b4bec6',lw=1)
 a.scatter(before[r['id']],i,color='#b4bec6',s=20,zorder=2)
 a.scatter(r['error_ev'],i,color='#007f80' if r['family']=='pi_pi' else '#72538c',s=42 if r['calibration'] else 25,marker='*' if r['calibration'] else 'o',zorder=3)
a.set(yticks=y,yticklabels=[r['molecule'].replace('_Intermediate','')+' '+r['id']+(' *' if r['calibration'] else '') for r in rows],ylim=(len(rows)-.5,-.7),xlim=(-.6,1.3),xlabel='Signed vertical error / eV');a.tick_params(axis='y',labelsize=11);a.grid(axis='x',alpha=.15);a.set_title('(a) All nine pi → pi* development states',loc='left',fontsize=12,fontweight='bold')
a.scatter([],[],c='#b4bec6',s=20,label='Original screened-even model');a.scatter([],[],c='#007f80',s=25,label='Corrected pi → pi*');handles,legend_labels=a.get_legend_handles_labels();f.legend(handles,legend_labels,loc='lower center',bbox_to_anchor=(.5,.065),ncol=2,fontsize=10,frameon=False)
eth=next(r for r in rows if r['id']=='B002');values=[9.078513469522075,eth['prediction_ev'],eth['TDA_ev']];labels=['Original OO-CID','Corrected OO-CID','Matched TDA'];colors=['#a7b2bc','#007f80','#d6e0e5'];b.axvspan(7.897-.5,7.897+.5,color='#eff5f5');b.axvline(7.897,color='#344e60',ls='--',lw=1)
for i,(value,col) in enumerate(zip(values,colors)):
 b.scatter(value,i*.8,s=65,color=col,edgecolors='#506575',linewidths=.6);b.text(value,i*.8-.12,f'{value:.3f} eV',ha='center',fontsize=11)
b.set(yticks=[0,.8,1.6],yticklabels=labels,ylim=(3.2,-.6),xlim=(7.25,11.3),xlabel='Ethylene vertical excitation / eV');b.tick_params(axis='y',labelsize=11);b.set_title('(b) Ethylene, native GFN2 orbital chart',loc='left',fontsize=12,fontweight='bold');b.text(.03,.04,'Reference: 7.897 eV\nCorrected error: +0.309 eV\n\n50/50 F/Q, lambda = 0.9\nZero energy offset\nOne calibrated parameter:\nbath efficiency eta = 0.60',transform=b.transAxes,fontsize=11,va='bottom',color='#344e60')
f.text(.5,.015,'Stars: two states used to calibrate eta. All 23 labels were exposed; this is not a blind benchmark.',ha='center',fontsize=9,color='#4b5f6e');f.tight_layout(rect=(0,.20,1,1),w_pad=2.1)
for ext in ['pdf','png']:f.savefig(P/'figures'/('ethylene_repair.'+ext),dpi=220,bbox_inches='tight')
lines=[r'\begin{longtable}{llrrrr}',r'\toprule Molecule / state & Family & Reference & Corrected OO & Matched TDA & OO error\\\midrule\endhead']
for r in allrows:lines.append(f"{r['molecule'].replace('_', ' ')} / {r['id']}"+(' $^*$' if r['calibration'] else '')+f" & {'$\\pi\\pi^*$' if r['family']=='pi_pi' else '$n\\pi^*$'} & {r['reference_ev']:.3f} & {r['prediction_ev']:.3f} & {r['TDA_ev']:.3f} & {r['error_ev']:+.3f}"+r'\\')
lines.append(r'\bottomrule\end{longtable}');(P/'tables/empirical_bath_vertical.tex').write_text('\n'.join(lines)+'\n')
