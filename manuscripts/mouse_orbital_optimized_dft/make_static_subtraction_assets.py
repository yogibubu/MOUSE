"""Plot scalar candidate evidence only; never run an electronic calculation."""
from pathlib import Path
import argparse,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
P=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--archive',required=True);a=parser.parse_args();archive=Path(a.archive)
scored=json.loads((archive/'small_molecules/scored_summary.json').read_text());parent=json.loads((P/'data/spectral_blocks/spectral_block_summary.json').read_text());rows=scored['rows'];assert len(rows)==26
completion=json.loads((archive/'public_models/completion.json').read_text());assert len(completion)==24 and all(r['returncode']==0 for r in completion)
certificates=[json.loads(p.read_text()) for p in (archive/'public_models').glob('*/*/runtime_certificate.json')];assert len(certificates)==24
for c in certificates:assert c['static_self_energy_subtraction'] and c['energy_max_error_ev']<1e-7 and max(abs(x) for x in c['spin_squared'])<1e-8 and c['restart_error_hartree']<1e-9
summary=scored['summary'];colors={'pi_pi':'#B96338','n_pi':'#087F88'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,(ax,bx)=plt.subplots(1,2,figsize=(10.2,3.9),gridspec_kw={'width_ratios':[1,1.8]},layout='constrained')
groups=['all','pi_pi','n_pi'];x=np.arange(3);width=.25
for i,(label,values,color) in enumerate([('Matched TDA',[parent['summary'][g]['TDA']['MAE_ev'] for g in groups],'#A4ABB0'),('Before matching',[parent['summary'][g]['MAE_ev'] for g in groups],'#D3BCA0'),('Static matching',[summary[g]['MAE_ev'] for g in groups],'#087F88')]):
 bars=ax.bar(x+(i-1)*width,values,width,color=color,label=label)
 for bar,value in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,value+.018,f'{value:.3f}',ha='center',va='bottom',fontsize=7,rotation=90)
ax.set(xticks=x,xticklabels=['All 26',r'12 $\pi\pi^*$',r'14 $n\pi^*$'],ylabel='Mean absolute error / eV',ylim=(0,1.8),title='A  Same orbitals and reference labels');ax.legend(frameon=False,fontsize=8);ax.grid(axis='y',alpha=.14);ax.set_axisbelow(True)
ordered=sorted(rows,key=lambda r:(r['family']!='pi_pi',r['id']))
for i,r in enumerate(ordered):
 before=abs(r['native_ev']-r['reference_ev']);after=abs(r['error_ev']);color=colors[r['family']]
 bx.plot([i,i],[before,after],color=color,lw=1.3,alpha=.7)
 bx.scatter(i,before,s=20,marker='s',facecolors='white',edgecolors='#7C858A',linewidths=.8,zorder=3)
 bx.scatter(i,after,s=26,color=color,zorder=4)
bx.axhline(.5,color='#A0A6A9',ls='--',lw=1);bx.axvline(11.5,color='#D4D8D9',lw=.8);bx.text(5.5,.55,r'$\pi\rightarrow\pi^*$',ha='center',color=colors['pi_pi']);bx.text(18.5,.55,r'$n\rightarrow\pi^*$',ha='center',color=colors['n_pi'])
bx.set(xticks=range(26),xticklabels=[r['id'] for r in ordered],ylabel='Absolute error / eV',ylim=(0,.6),title='B  Every exposed development state');bx.tick_params(axis='x',rotation=90,labelsize=7);bx.grid(axis='y',alpha=.14)
bx.scatter([],[],s=20,marker='s',facecolors='white',edgecolors='#7C858A',label='Before matching');bx.scatter([],[],s=26,color='#087F88',label='Static matching');bx.legend(frameon=False,fontsize=8,loc='upper right')
fig.savefig(P/'figures/static_subtraction_accuracy.pdf',bbox_inches='tight');fig.savefig(P/'figures/static_subtraction_accuracy.png',dpi=200,bbox_inches='tight');plt.close(fig)
data=dict(scope='Exposed development candidate; no independent validation or universal-accuracy claim',weights=[.5,.5],lambda_exchange=.9,eta=.6,energy_offset_ev=0,summary=summary,rows=rows,public_API=dict(molecules=24,states=26,maximum_energy_replay_error_ev=max(c['energy_max_error_ev'] for c in certificates),maximum_restart_error_hartree=max(c['restart_error_hartree'] for c in certificates),maximum_spin_squared=max(abs(x) for c in certificates for x in c['spin_squared']),DFT_imports=sum(c['DFT_imports'] for c in certificates),new_SCC_calls=sum(c['new_SCC_calls'] for c in certificates),TDA_solves=sum(c['TDA_solves'] for c in certificates)))
(P/'data/static_subtraction_summary.json').write_text(json.dumps(data,indent=2)+'\n')
lines=[r'\begin{table}[!htbp]\centering\small',r'\caption{Static-matching development candidate on the same corrected native orbitals, geometries and 26 exposed reference labels. No parameter is refitted. All errors are in eV.}',r'\label{tab:staticmatching}',r'\begin{tabular}{lrrrrr}\toprule',r'Family & States & TDA MAE & Before MAE & Matched MAE & Matched max.\\\midrule']
for label,g in [('All','all'),(r'$\pi\pi^*$','pi_pi'),(r'$n\pi^*$','n_pi')]:
 s=summary[g];lines.append(f"{label} & {s['count']} & {parent['summary'][g]['TDA']['MAE_ev']:.3f} & {parent['summary'][g]['MAE_ev']:.3f} & {s['MAE_ev']:.3f} & {s['maximum_error_ev']:.3f}\\\\")
lines.extend([r'\bottomrule\end{tabular}',r'\end{table}']);(P/'tables/static_subtraction_summary.tex').write_text('\n'.join(lines)+'\n')
